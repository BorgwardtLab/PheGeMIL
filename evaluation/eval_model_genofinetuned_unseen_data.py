# ------------------------------------------
# 1.2021
# Evaluation script for unseen data using MIL with geno-only finetuning
# Copied the other script and adapted it for better traceability.
# Highly similar to eval_deepgeno_baseline
#
# Generate predictions for new dataset with only genotypes
# The script fetches the model that was hyperoptimized via random search.
# ------------------------------------------
import os
import numpy as np
import pandas as pd
import tempfile
import json
from tqdm import tqdm

from sacred import Experiment
from sacred.observers import FileStorageObserver

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
ex = Experiment("eval_new_unseen_set_MIL_genofinetuned_baseline")

# Import models from historical versions
from src.modules import *
from src.genotype_MIL_finetuning_adv import *
from src.genotype_deep_baseline import *
from src.data_test import FusedFullBags, variable_length_collate_fullfusedbags, GenotypeDataset
from types import MethodType

@ex.config
def cfg():
    split = 0
    run = {
        'df'             : 'df_20201110_numpy_MIL_npy_coordinates_geno_filtered.csv',
        'df_out'         : f'df_20210124_predictions_norm_target_model_genofinetuned_split{split}.csv',
        'prefix'         : 'unseen_data_evaluation_model_genofinetuned',
        'use_gpu'        : False,
        'split'          : split,

    }
    dirs = {
        'data_folder'    : '/links/groups/borgwardt/Data/Jesse_2018/202010_new_testset',
        'output_folder'  : '/links/groups/borgwardt/Data/Jesse_2018/output/202010_new_testset',
        'genotype'       : '/links/groups/borgwardt/Data/Jesse_2018/numpy_MIL_resized/genotypes',
        'trained_models' : f'exp_logs/20210124_genotype_mil_finetuning/version_{split}',
        'target_standardize_path' : '/links/groups/borgwardt/Data/Jesse_2018/csv/df_20201125_normalization_targets.csv',
        'genotype_scaler_path' : f'/links/groups/borgwardt/Data/Jesse_2018/csv/genotype_scalers/new_scaler_fold{split}.pkl',
    }

ex.observers.append(FileStorageObserver.create("/links/groups/borgwardt/Data/Jesse_2018/output/logs/sacred_logs/202011_newtestset_eval_deepgeno"))

# Load trained model
def get_saved_checkpoint(folder):
    # Manual extraction of the epoch number for the
    for f in os.listdir(folder):
        if os.path.splitext(f)[1] == '.ckpt':
            return f

# Special geno-only fwd
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


def load_trained_model(folder, on_gpu=False):
    # load the model from its latest checkpoint
    model = MILCropYieldGenoFinetune.load_from_metrics(
                weights_path=os.path.join(folder,get_saved_checkpoint(folder)),
                tags_csv=os.path.join(folder, 'meta_tags.csv'),
                on_gpu=on_gpu)
    model.model.forward_genotype = MethodType(forward_genotype, model.model)
    return model


def eval_model_regression(model, dataloader, use_gpu=False):
    model.train(False)  # Set model to evaluate mode
    model.eval()

    # Collect predictions in validation
    true_y = []
    pred_y = []

    # Iterate over data.
    for data in tqdm(dataloader):
        # 1. explicit forward get the inputs
        data_genotype, labels = data
        if use_gpu:
            data_genotype = data_genotype.cuda()
        y_pred = model.forward_genotype(data_genotype)
           
        batch_size = y_pred.size()[0]
        y_pred = y_pred.squeeze()
        if  batch_size == 1: # need to do this to ensure good dimensionalities for batch size == 1
            y_pred = y_pred.unsqueeze(0)


        # Transform to numpy to unload torch memory.
        if use_gpu:
            y_pred = y_pred.cpu()
        pred_y.append(y_pred.detach().data.numpy())
        true_y.append(labels.detach().numpy())

    # Unload the values from the gpu
    y_true = np.concatenate(true_y, axis=None).reshape(-1)
    y_pred = np.concatenate(pred_y, axis=None).reshape(-1)
    # torch.cat(true_y).t().data.cpu().numpy().reshape(-1)

    results = dict()
    results['mse'] = mean_squared_error(y_true, y_pred)
    results['mae'] = mean_absolute_error(y_true, y_pred)
    results['r2'] = r2_score(y_true, y_pred)
    results['pearson'] = pearsonr(y_true,y_pred)[0]

    print('MSE: {:.4f}, R2_SCORE: {:.4f}, Pearson {:.4f}'.format(results['mse'], results['r2'], results['pearson']))

    return y_true, y_pred, results

@ex.automain
def main(_rnd, run, dirs):
    ###########################
    # Load initial config file and files
    ###########################
    # Initiate results
    results_cross_fold = {}
    # print(config)

    # Load datasets
    base_path = os.path.join(dirs['data_folder'], 'numpy_MIL_resized')
    # TODO: change the file loader here.

    df_test = pd.read_csv(os.path.join(dirs['data_folder'], 'csv', run['df']))
    # Latest_time is the latest_date for which we want to keep information
    ds = GenotypeDataset(os.path.join(dirs['data_folder'], 'csv', run['df']),
                base_path=dirs['genotype'],
                normalize=True,
                split_id=run['split'], # A single json containing all samples in test split 0
                scaler_path=dirs['genotype_scaler_path'],
                split_name='test',
                standardize_target_path=dirs['target_standardize_path'],
                )

    dataloader=DataLoader(ds,
                batch_size=16,
                shuffle=False, # Keep order
                num_workers=10, #1 if run['use_gpu'] else 8, # 1 for CUDA
                pin_memory=run['use_gpu'] # CUDA only
                )
    print("Data loaded")

    # run through the different trained models
    for fold in range(1):
        print("Computing results for trained model {}.".format(fold))
        # Load model
        folder = os.path.join(dirs['trained_models'])#Used to be hardcoded, f'version_1') # HARDCODED
        model = load_trained_model(folder, on_gpu=run['use_gpu'])
        print("Model loaded.")
        # GPU
        if run['use_gpu']:
            # Transfer the model on the GPU
            model.cuda()

        y_true, y_pred, results = eval_model_regression(model.model, dataloader,
                    use_gpu=run['use_gpu'])

        # Add predictions to df and save
        df_test['PREDICTED_YIELD'] = y_pred
        df_test['ACTUAL_YIELD_(VALID)'] = y_true

        # Save model and results
        results_cross_fold[fold]=results

        # Save to file
        path_fold = os.path.join(dirs['output_folder'],'imputation', str(fold))
        if not os.path.exists(path_fold):
            os.makedirs(path_fold)

        df_test.to_csv(os.path.join(path_fold, run['df_out']), index=False)

        # Copy in sacred logs
        ex.add_artifact(os.path.join(path_fold, run['df_out']), name='test_fold{}.csv'.format(fold))

        if run['use_gpu']:
            # Delete referenced variables and free GPU cache
            del model
            torch.cuda.empty_cache()
        print()
        print()

    return results_cross_fold