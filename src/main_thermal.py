"""
Contains the main pytorch lightning scripts

M. Togninalli, 11.2019
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import tensor

from test_tube import HyperOptArgumentParser

from .modules import Aggregator, DeepMultiFusionMIL
from .util import run_fitting_experiment

import os
from collections import OrderedDict

import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from sklearn.metrics import r2_score

from .data import FusedBags, CropTransforms, variable_length_collate_fusedbags, get_transformations

import random


###########
# UTILS
##########
def add_general_arguments(parser, default_experiment_name):
    group = parser.add_argument_group('training')
    group.add_argument('--max-epochs', default=100, type=int)
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



class MILCropYieldThermal(pl.LightningModule):
    """Base class for a the MILCropYieldThermal model
    Implements parsing common arguments, loading of dataset and basic training
    routine.
    """

    def __init__(self, hparams):
        """Store parameters as self.hparams and call __build_model()."""
        # init superclass
        super().__init__()
        self.hparams = hparams

        # build model
        self.model = DeepMultiFusionMIL(multispectral=True, 
                    encoder_type_multispectral=self.hparams.encoder_type, 
                    thermal=True, encoder_type_thermal=self.hparams.encoder_type_thermal,
                    thermal_image_size=(40,40), heads=self.hparams.n_heads, 
                    concat_or_aggregate=self.hparams.concat_or_aggregate, 
                    temporal_encoding=self.hparams.temporal_encoding, n_dates=16,
                    channel_encoding=self.hparams.channel_encoding)
    
    def loss(self, labels, model_output):
        """Compute the loss."""
        # Alternative SmoothL1Loss
        mse = F.mse_loss(model_output, labels)
        return mse

    def training_step(self, data_batch, batch_i):
        """Run a single training step."""
        # forward pass
        if self.hparams.temporal_encoding:
            # Also receives dates
            data_multispectral, data_thermal, labels, lengths_multispectral, \
                lengths_thermal, dates_multispectral, dates_thermal = data_batch
            y_pred = self.forward((data_multispectral, lengths_multispectral, data_thermal, \
                                    lengths_thermal,dates_multispectral, dates_thermal))
        else:
            data_multispectral, data_thermal, labels, lengths_multispectral, lengths_thermal = data_batch
            y_pred = self.forward((data_multispectral, lengths_multispectral, data_thermal, lengths_thermal))
        batch_size = y_pred.size()[0]
        y_pred = y_pred.squeeze()
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
        if self.hparams.temporal_encoding:
            # Also receives dates
            data_multispectral, data_thermal, labels, lengths_multispectral, \
                lengths_thermal, dates_multispectral, dates_thermal = data_batch
            y_pred = self.forward((data_multispectral, lengths_multispectral, data_thermal, \
                                    lengths_thermal,dates_multispectral, dates_thermal))
        else:
            data_multispectral, data_thermal, labels, lengths_multispectral, lengths_thermal = data_batch
            y_pred = self.forward((data_multispectral, lengths_multispectral, data_thermal, lengths_thermal))
        batch_size = y_pred.size()[0]
        y_pred = y_pred.squeeze()
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
        # Use pre-determined splits
        transforms = None
        size = (128,128) # if self.hparams.encoder_type == 'convnet' else (224,224)
        # Add path information
        fixed_size = not 'df_20190814_numpy_MIL_coordinates' in self.hparams.csv_name
        if split == 'train':
            transforms = get_transformations(augmentation=self.hparams.augmentation, size=size, fixed_size=fixed_size)
        else:
            transforms = get_transformations(augmentation='none', size=size, fixed_size=fixed_size)
        tmp_folder = 'matteos_cached_resized_imgs'
        if fixed_size:
            base_path_multispectral = os.path.join('/tmp', tmp_folder, '2017-2018_CIMMYT_Wheat')
            base_path_thermal = os.path.join('/tmp', tmp_folder, 'thermal_images')
        else:
            base_path_multispectral = os.path.join(self.hparams.data_path,'numpy_MIL', '2017-2018_CIMMYT_Wheat')
        
        # Ensure dates are present
        if self.hparams.temporal_encoding:
            assert 'df_20191121_numpy_MIL_npy_coordinates_dates' in self.hparams.csv_name
        # base_path_thermal = os.path.join(self.hparams.data_path,'numpy_MIL_resized', 'thermal_images') #This is the original path
        dataset = FusedBags(os.path.join(self.hparams.data_path,'csv', self.hparams.csv_name),
                base_path_multispectral=base_path_multispectral,
                base_path_thermal=base_path_thermal,
                # base_path=os.path.join(self.hparams.data_path,'numpy_MIL_resized', '2017-2018_CIMMYT_Wheat'),
                # base_path=os.path.join('/tmp',tmp_folder, '2017-2018_CIMMYT_Wheat'),
                # transform=CropTransforms(transforms), # CAREFUL: if resized is True, no transforms will be applied.
                subsample_size=self.hparams.bag_size,
                split_id=self.hparams.split_id,
                split_name=split,
                resized=fixed_size,
                return_dates=self.hparams.temporal_encoding
                )
        print(f'Returning {split} with {len(dataset)} instances.')
        return dataset

    def __dataloader(self, split):
        dataset = self.get_dataset(split)

        loader = DataLoader(
            dataset=dataset,
            collate_fn=variable_length_collate_fusedbags,
            batch_size=self.hparams.batch_size,
            shuffle=split == 'train',
            # When using caching we can onlyu use a single worker
            # num_workers=0 if self.hparams.cache_dataset else 8, # 1 for CUDA?
            num_workers=12, # 1 for CUDA?
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
    
    def forward(self, x_tuple):
        """Compute the forward pass."""
        # x_tuple contains the four elements
        x = self.model(x_tuple)
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
            '--csv-name', default='df_20191014_numpy_MIL_npy_coordinates.csv', type=str) # df_20190927_numpy_MIL_npy_coordinates.csv
        group.add_argument(
            '--normalize', default=False, action='store_true',
            help='Normalize persistence diagrams using mean and std.')
        group.add_argument(
            '--cv-round', default=0, type=int,
            help='File in data root to load for split definitions.'
        )
        group.add_argument('--split-id', default=0, type=int)

        # Temporal encoding
        parent_parser.opt_list('--augmentation', default='light', type=str,
                    options=['light', 'strong'])
        parent_parser.opt_list(
            '--temporal-encoding', default=False, action='store_true', tunable=True, options=[True, False],
            help='Encode dates of the images.'
        )
        parent_parser.opt_list(
            '--channel-encoding', default=False, action='store_true', tunable=True, options=[True, False],
            help='Encode data channels (multispectral vs thermal).'
        )
        return parent_parser

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        # Training options
        parent_parser.opt_list('--bag-size', default=None, type=int,
                        options=[32, 16, 8], tunable=True,
                        help='Bag size is the maximum number of images per sample '
                        'across all nodes')
        parent_parser.opt_list('--batch-size', default=16, type=int,
                        options=[8, 16], tunable=True,
                        help='batch size will be divided over all gpus being used '
                        'across all nodes')
        parent_parser.opt_list('--learning-rate', default=0.0001, type=float,
                        options=[0.000001, 0.00001, 0.0001, 0.001],
                        tunable=True)


        # Model options (Encoder, aggregator)

        parent_parser.add_argument('--encoder-type', default='resnet', type=str,
                        choices=['convnet', 'resnet'])
        parent_parser.opt_list('--encoder-type-thermal', default='convnet', type=str,
                        options=['convnet', 'resnet'], tunable=True)
        parent_parser.opt_list('--n-heads', default=1, type=int,
                        options=[1, 4, 8],
                        tunable=True)
        parent_parser.opt_list('--scheduler', default='none', type=str,
                        options=['none', 'cosine', 'plateau'], #, 'plateau'
                        tunable=True)
        parent_parser.add_argument('--concat-or-aggregate', default='aggregate', type=str,
                        choices=['aggregate', 'concat'], # Defines whether the representations
                        # are aggregated or concatenated (see modules)
                        )

        parser = cls.add_dataset_specific_args(parent_parser)

        return parser


def get_parser():
    parent_parser = HyperOptArgumentParser(
        strategy='random_search', add_help=False)

    # Define and parse arguments
    parent_parser = add_general_arguments(parent_parser, 'thermal_hypersearch')
    parser = MILCropYieldThermal.add_model_specific_args(parent_parser)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    random.seed(42)
    hyperparams = parser.parse_args()

    if hyperparams.hypersearch:
        for trial in hyperparams.trials(hyperparams.n_trials):
            print(trial)
            run_fitting_experiment(MILCropYieldThermal, trial)
    else:
        run_fitting_experiment(MILCropYieldThermal, hyperparams)
