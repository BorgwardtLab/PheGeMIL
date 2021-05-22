"""
Contains the script for the fine-tuning of the 
genotype model
M. Togninalli, 12.2020
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import tensor

from test_tube import HyperOptArgumentParser

from .main_geno import MILCropYieldGeno
from .util import run_fitting_experiment

import os
from collections import OrderedDict

import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from sklearn.metrics import r2_score

from .data import GenotypeDataset
from src.modules import *
from src.main_dem_geno import *

import random, csv
from types import MethodType

###########
# UTILS
##########
def add_general_arguments(parser, default_experiment_name):
    group = parser.add_argument_group('training')
    group.add_argument('--max-epochs', default=10, type=int) # 10 by default, 1 recommended
    group.add_argument('--gpus', type=str, default='0')
    group.add_argument('--hypersearch', action='store_true', default=False)
    group.add_argument('--n-trials', default=15, type=int) # 8
    # group.add_argument('--hypersearch_gpus', default=1, type=int) # 8

    # Logging settings
    group = parser.add_argument_group('logging')
    default_log_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__), '..', 'exp_logs'))
    group.add_argument('--log-dir', default=default_log_dir)
    group.add_argument('--exp-name', default=default_experiment_name)
    group.add_argument('--exp-version', default=None, type=int)
    group.add_argument('--debug', action='store_true', default=False,
                       help='Treat as debug run, dont write anything to log')
    return parser


# Load trained model
def get_saved_checkpoint(folder):
    # Manual extraction of the epoch number for the
    for f in os.listdir(folder):
        if os.path.splitext(f)[1] == '.ckpt':
            return f

def load_trained_model(folder, on_gpu=False):
    # load the model from its latest checkpoint
    model = MILCropYieldFull.load_from_metrics(
                weights_path=os.path.join(folder,get_saved_checkpoint(folder)),
                tags_csv=os.path.join(folder, 'meta_tags.csv'),
                on_gpu=on_gpu)
    return model

class MILCropYieldGenoFinetune(pl.LightningModule):
    """Fine tuning class for a the MILCropYieldGeno model
    Implements parsing common arguments, loading of dataset and basic training
    routine.
    """

    def __init__(self, hparams):
        """Store parameters as self.hparams and call __build_model()."""
        # init superclass
        super().__init__()
        self.hparams = hparams

        # build model
        def get_saved_checkpoint(folder):
            # Manual extraction of the epoch number for the
            for f in os.listdir(folder):
                if os.path.splitext(f)[1] == '.ckpt':
                    return f

        # Build original folder
        self.original_folder = f'{self.hparams.exp_path}/{self.hparams.original_experiment}/version_{self.hparams.original_version}'

        # Ensure the correct splits are selected
        fname = os.path.join(self.original_folder, 'meta_tags.csv')
        with open(fname, mode='r') as infile:
            reader = csv.reader(infile)
            for rows in reader:
                if rows[0] == 'split_id':
                    self.hparams.split_id = int(rows[1])
                    print(f'The correct split id is {self.hparams.split_id}')

        # Get GPUs config 
        gpus = str(hparams.gpus) if hparams.gpus != '' else None
        if gpus is not None:
            # gpu is a string, thus this should also work with a singe gpu
            if len(gpus) > 1:
                gpus = gpus.split(',')[0]

        model1 = load_trained_model(self.original_folder, on_gpu=gpus is not None)

        # Only keep the genotype part
        def forward_genotype(self, x_genotype):
            x_genotype = self.genotype_encoder(x_genotype)

            if self.temporal_encoding:
                x_genotype = torch.cat([x_genotype, torch.zeros(x_genotype.size()[0],16, device=x_genotype.device)], dim=1)    
            if self.channel_encoding:
                # Add encoding for each channel
                x_genotype = self.pad_1(self.pad_0(x_genotype))
            # Must align x_genotype with format of bags batch_size x 1 x embedding_dim
            x_genotype = x_genotype.unsqueeze(1)
            x = torch.cat([x_genotype], dim=1)
            lengths = [torch.ones(len(x_genotype), dtype=torch.int64)]

            # Aggregate
            x = self.aggregator(x, lengths)

            batch_size, n_heads, L = x.size()
            x = x.view(batch_size, L*n_heads)
            return self.regressor(x)

        model1.model.fwd_genotype = MethodType(forward_genotype, model1.model)
        self.model = model1.model

        # Freeze the encoder, if required
        if self.hparams.freeze_encoder:
            for param in self.model.genotype_encoder.parameters():
                param.requires_grad = False
            
    
    def loss(self, labels, model_output):
        """Compute the loss."""
        # Alternative SmoothL1Loss
        mse = F.mse_loss(model_output, labels)
        return mse

    def training_step(self, data_batch, batch_i):
        """Run a single training step."""
        # forward pass
        data_genotype, labels = data_batch
        y_pred = self.forward(data_genotype)
        batch_size = y_pred.size()[0]
        # y_pred = y_pred.squeeze()
        if  batch_size == 1: # need to do this to ensure good dimensionalities for batch size == 1
            y_pred = y_pred.unsqueeze(0)

        # calculate loss
        loss_val = self.loss(labels, y_pred)

        # in DP mode (default) make sure if result is scalar, there's another
        # dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, data_batch, batch_i):
        """Run a single validation step."""
        # forward pass
        data_genotype, labels = data_batch
        y_pred = self.forward(data_genotype)
        batch_size = y_pred.size()[0]
        # y_pred = y_pred.squeeze()
        if  batch_size == 1: # need to do this to ensure good dimensionalities for batch size == 1
            y_pred = y_pred.unsqueeze(0)
        output = OrderedDict({
            'labels': labels,
            'predictions': y_pred,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """Aggregate outputs from validation."""
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_mae_mean = 0
        total_samples = 0
        # Simply recompute 
        labels = []
        predictions = []
        for output in outputs:
            labels.append(output['labels'])
            predictions.append(output['predictions'])

        labels = torch.cat(labels)
        predictions = torch.cat(predictions)

        loss_val = self.loss(labels, predictions)

        # Compute other metrics MSE, r2, pearson
        val_mae = torch.mean(torch.abs(labels-predictions))
        val_r2 = r2_score(labels.cpu().numpy(), predictions.cpu().numpy())

        if self.on_gpu:
            val_mae = val_mae.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
            val_mae = val_mae.unsqueeze(0)
        # For plateau scheduler
        self.val_loss = loss_val

        tqdm_dic = {'val_loss': loss_val, 'val_mae': val_mae, 'val_r2': val_r2}
        return tqdm_dic

    def on_post_performance_check(self):
        # Called after the validation loop
        if self.hparams.scheduler == 'plateau':
            # call scheduler
            self.scheduler.step(self.val_loss)
            del self.val_loss

    def configure_optimizers(self):
        # Check if LR scheduler
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.scheduler == 'none':
            return self.optimizer
        elif self.hparams.scheduler == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer)
            return self.optimizer
        elif self.hparams.scheduler == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.hparams.max_epochs)
            return [self.optimizer], [self.scheduler]

    def get_dataset(self, split):
        # Add path information        
        tmp_folder = '/links/groups/borgwardt/Data/Jesse_2018/numpy_MIL_resized'
        base_path_genotype = os.path.join(tmp_folder, 'genotypes')

        # Normalize targets
        if self.hparams.normalize_targets:
            standardize_target_path = os.path.join(self.hparams.data_path,'csv', 'df_20201125_normalization_targets.csv')
        else:
            standardize_target_path = None

        dataset = GenotypeDataset(os.path.join(self.hparams.data_path,'csv', self.hparams.csv_name),
                base_path=base_path_genotype,
                normalize=self.hparams.normalize_genotype, # Look at historical parameters
                split_id=self.hparams.split_id,
                split_name=split,
                standardize_target_path=standardize_target_path,
                scaler_path=self.hparams.genotype_scaler_path,
                )
        print(f'Returning {split} with {len(dataset)} instances.')
        return dataset

    def __dataloader(self, split):
        dataset = self.get_dataset(split)

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=split == 'train',
            # When using caching we can onlyu use a single worker
            # num_workers=0 if self.hparams.cache_dataset else 8, # 1 for CUDA?
            num_workers=10, # 1 for CUDA?
            pin_memory=self.on_gpu
        )
        return loader

    @pl.data_loader
    def tng_dataloader(self):
        return self.__dataloader(split='train')

    @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader(split='train')

    @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader(split='val')

    @pl.data_loader
    def test_dataloader(self):
        return self.__dataloader(split='test')
    
    def forward(self, x):
        """Compute the forward pass."""
        x = self.model.fwd_genotype(x)
        return x

    @classmethod
    def add_dataset_specific_args(cls, parent_parser):  # pragma: no cover
        """Just add parameters relevant for our type of data.
        Args:
            parent_parser:
        Returns: A parser with options regarding the dataset augmented.
        """
        # data
        group = parent_parser.add_argument_group('dataset')
        group.add_argument(
            '--data-path', default='/links/groups/borgwardt/Data/Jesse_2018', type=str)
        group.add_argument(
            '--csv-name', default='df_20200107_numpy_MIL_npy_coordinates_dates_genofiltered.csv', type=str) # df_20190927_numpy_MIL_npy_coordinates.csv
        group.add_argument('--split-id', default=0, type=int)
        
        return parent_parser

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        # Version
        parent_parser.add_argument('--exp-path', default='/home/tomatteo/Projects/yield_prediction/code/exp_logs', type=str)
        parent_parser.add_argument('--original-experiment', default='20201125_normalized_training_ms_geno_normalized_target', type=str)
        parent_parser.add_argument('--original-version', default='1', type=str)

        # Training options
        parent_parser.opt_list('--batch-size', default=16, type=int,
                        options=[16, 32, 64], tunable=True,
                        help='batch size will be divided over all gpus being used '
                        'across all nodes')
        parent_parser.opt_list('--learning-rate', default=0.0001, type=float,
                        options=[0.000001, 0.00001, 0.0001, 0.001],
                        tunable=True)
        parent_parser.add_argument('--normalize-targets', default=False, action='store_true',
                    help='Normalize target labels using mean and std.')


        # Model options (Encoder, aggregator)
        parent_parser.opt_list('--normalize-genotype', default=False, 
                    action='store_true', tunable=True, options=[True, False])
        parent_parser.opt_list('--genotype-scaler-path', default=None, # '/links/groups/borgwardt/Data/Jesse_2018/csv/genotype_scalers/scaler_fold0.pkl',
                    tunable=True)
        parent_parser.opt_list('--freeze-encoder', default=False, 
                    action='store_true', tunable=True, options=[True, False])
        parent_parser.opt_list('--scheduler', default='none', type=str,
                        options=['none', 'cosine', 'plateau'], #, 'plateau'
                        tunable=True)
        parser = cls.add_dataset_specific_args(parent_parser)

        return parser


def get_parser():
    parent_parser = HyperOptArgumentParser(
        strategy='random_search', add_help=False)

    # Define and parse arguments
    parent_parser = add_general_arguments(parent_parser, '20210124_genotype_mil_finetuning')
    parser = MILCropYieldGenoFinetune.add_model_specific_args(parent_parser)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    random.seed(42)
    hyperparams = parser.parse_args()

    if hyperparams.hypersearch:
        for trial in hyperparams.trials(hyperparams.n_trials):
            print(trial)
            run_fitting_experiment(MILCropYieldGenoFinetune, trial)
    else:
        run_fitting_experiment(MILCropYieldGenoFinetune, hyperparams)
