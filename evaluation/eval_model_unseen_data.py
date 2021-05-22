# ------------------------------------------
# 10.2020
# Evaluation script for unseen data 
#
# Generate predictions for new dataset without
# genotype nor thermal images.
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
ex = Experiment("eval_new_unseen_set")

# Import models from historical versions
# from srctmp_20200211.data import *
# from srctmp_20200211.modules import *
# from srctmp_20200211.main_dem_geno import *
from src.modules import *
from src.main_dem_geno import *
from src.data_test import FusedFullBags, variable_length_collate_fullfusedbags
from types import MethodType

@ex.config
def cfg():
    split = 1
    run = {
        'df'             : 'df_20201110_numpy_MIL_npy_coordinates_geno_filtered.csv',
        'df_out'         : f'df_20201230_predictions_norm_target_halved_images_split{split}.csv',
        'prefix'         : 'unseen_data_evaluation_MIL_ms_geno',
        'use_gpu'        : True,
        'only_genotype'  : False,
        'split'          : split,

    }
    dirs = {
        'data_folder'    : '/links/groups/borgwardt/Data/Jesse_2018/202010_new_testset',
        'output_folder'  : '/links/groups/borgwardt/Data/Jesse_2018/output/202010_new_testset',
        'multispectral'  : 'multispectral_images_Or_large_crop',
        'thermal'        : None,
        'genotype'       : '/links/groups/borgwardt/Data/Jesse_2018/numpy_MIL_resized/genotypes',
        'dem'            : None,
        'trained_models' : f'exp_logs/20201125_normalized_training_ms_geno_halved_images_normalized_target/version_{split+1}',
        'standardize_path' : '/links/groups/borgwardt/Data/Jesse_2018/csv/df_20201108_normalization_features.csv',
        'target_standardize_path' : '/links/groups/borgwardt/Data/Jesse_2018/csv/df_20201125_normalization_targets.csv',
        'genotype_scaler_path' : f'/links/groups/borgwardt/Data/Jesse_2018/csv/genotype_scalers/scaler_fold{split}.pkl',
    }

ex.observers.append(FileStorageObserver.create("/links/groups/borgwardt/Data/Jesse_2018/output/logs/sacred_logs/202011_newtestset_eval_ms_geno"))

# Load trained model
def get_saved_checkpoint(folder):
    # Manual extraction of the epoch number for the
    for f in os.listdir(folder):
        if os.path.splitext(f)[1] == '.ckpt':
            return f


def special_forward(self, x_multispectral=None, lengths_multispectral=None, x_thermal=None, lengths_thermal=None, x_dem=None, lengths_dem=None,
            x_genotype=None, dates_multispectral=None, dates_thermal=None, dates_dem=None):
    # TODO: to save on computation, avoid encoding empty images.
    if self.multispectral:
        x_multispectral = self.multispectral_encoder(x_multispectral)
        if self.temporal_encoding:
            x_multispectral = torch.cat([x_multispectral,dates_multispectral], dim=2)
        if self.channel_encoding:
            x_multispectral = self.pad_0(self.pad_1(x_multispectral))
    if self.thermal:
        x_thermal = self.thermal_encoder(x_thermal)
        if self.temporal_encoding:
            x_thermal = torch.cat([x_thermal,dates_thermal], dim=2)
        # if self.channel_encoding:
        #     # Add encoding for each channel
        #     x_thermal = self.pad_0(self.pad_0(self.pad_1(self.pad_0(x_thermal))))
    if self.dem:
        x_dem = self.dem_encoder(x_dem)
        if self.temporal_encoding:
            x_dem = torch.cat([x_dem,dates_dem], dim=2)
        # if self.channel_encoding: # Manual fix for ms and geno only
        #     # Add encoding for each channel
        #     x_dem = self.pad_0(self.pad_1(self.pad_0(self.pad_0(x_dem))))
    if self.genotype:
        x_genotype = self.genotype_encoder(x_genotype)
        if self.temporal_encoding:
            x_genotype = torch.cat([x_genotype, torch.zeros(x_genotype.size()[0],16, device=x_genotype.device)], dim=1)    
        if self.channel_encoding:
            # Add encoding for each channel
            x_genotype = self.pad_1(self.pad_0(x_genotype))
    # Attn-Aggregate all together
    # Combine the embeddings --> Always use the same channel encoding
    if self.multispectral and self.thermal and self.dem and self.genotype:
        # Must align x_genotype with format of bags batch_size x 1 x embedding_dim
        x_genotype = x_genotype.unsqueeze(1)
        x = torch.cat([x_multispectral, x_thermal, x_dem, x_genotype], dim=1)
        # Add genotype length for masking
        lengths_genotype = torch.ones_like(lengths_dem)
        lengths = [lengths_multispectral, lengths_thermal, lengths_dem, lengths_genotype]
    elif self.multispectral and self.dem and self.genotype and not self.thermal: # No thermal
        # Must align x_genotype with format of bags batch_size x 1 x embedding_dim
        x_genotype = x_genotype.unsqueeze(1)
        x = torch.cat([x_multispectral, x_dem, x_genotype], dim=1)
        # Add genotype length for masking
        lengths_genotype = torch.ones_like(lengths_dem)
        lengths = [lengths_multispectral, lengths_dem, lengths_genotype]
    elif self.multispectral and self.dem and not self.thermal and not self.genotype: # No thermal nor genotype
        x = torch.cat([x_multispectral, x_dem], dim=1)
        # Add genotype length for masking
        lengths_genotype = torch.ones_like(lengths_dem)
        lengths = [lengths_multispectral, lengths_dem]
    elif self.multispectral and not self.dem and not self.thermal and not self.genotype: # Only MS
        x = torch.cat([x_multispectral], dim=1)
        # Add genotype length for masking
        lengths = [lengths_multispectral]
    elif self.genotype:
        x_genotype = x_genotype.unsqueeze(1)
        x = torch.cat([x_genotype], dim=1)
        lengths = [torch.ones_like(lengths_multispectral)]
    else:# DEPRECATED
        raise Warning("Incomplete code for less channels.")
    # Aggregate
    x = self.aggregator(x, lengths)

    batch_size, n_heads, L = x.size()
    x = x.view(batch_size, L*n_heads)
    return self.regressor(x)

def load_trained_model(folder, on_gpu=False):
    # load the model from its latest checkpoint
    model = MILCropYieldFull.load_from_metrics(
                weights_path=os.path.join(folder,get_saved_checkpoint(folder)),
                tags_csv=os.path.join(folder, 'meta_tags.csv'),
                on_gpu=on_gpu)
    model.model.special_forward = MethodType(special_forward, model.model)
    return model


def eval_model_regression(model, dataloader, use_gpu=False, only_genotype=False):
    model.train(False)  # Set model to evaluate mode
    model.eval()
    
    # Collect predictions in validation
    true_y = []
    pred_y = []

    # Iterate over data.
    for data in tqdm(dataloader):
        # 1. explicit forward get the inputs
        if model.temporal_encoding:
            # Also receives dates
            # TODO: modify here for future ref.
            data_multispectral, labels, lengths_multispectral, dates_multispectral = data
            y_pred = model.special_forward(x_multispectral=data_multispectral, lengths_multispectral=lengths_multispectral,
                                        dates_multispectral=dates_multispectral)
        else:
            # data_multispectral, labels, lengths_multispectral = data
            # if use_gpu:
            #     data_multispectral = data_multispectral.cuda()
            #     lengths_multispectral = lengths_multispectral.cuda()
            # y_pred = model.special_forward(x_multispectral=data_multispectral, lengths_multispectral=lengths_multispectral)
            data_multispectral, data_thermal, data_dem, data_genotype, labels, lengths_multispectral, lengths_thermal, \
                lengths_dem = data
            if use_gpu:
                data_multispectral = data_multispectral.cuda()
                lengths_multispectral = lengths_multispectral.cuda()
                data_thermal = data_thermal.cuda()
                lengths_thermal = lengths_thermal.cuda()
                data_dem = data_dem.cuda()
                lengths_dem = lengths_dem.cuda()
                data_genotype = data_genotype.cuda()
            if only_genotype:
                y_pred = model.special_forward(x_genotype=data_genotype, 
                            lengths_multispectral=lengths_multispectral)
            else:    
                y_pred = model.forward((data_multispectral, lengths_multispectral, data_thermal, lengths_thermal, data_dem,\
                    lengths_dem, data_genotype)) # no special forward needed

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
    # sacred_path = os.path.join(dirs['output_folder'], 'logs/sacred_logs/crops_MIL', str(run['training_id']))
    # with open(os.path.join(sacred_path, 'config.json'), 'r') as c:
    #     config = json.load(c)
    # Initiate results
    results_cross_fold = {}
    # print(config)

    # Load datasets
    base_path = os.path.join(dirs['data_folder'], 'numpy_MIL_resized')
    # TODO: change the file loader here.

    df_test = pd.read_csv(os.path.join(dirs['data_folder'], 'csv', run['df']))
    # Latest_time is the latest_date for which we want to keep information
    ds = FusedFullBags(os.path.join(dirs['data_folder'], 'csv', run['df']),
                base_path_multispectral=os.path.join(base_path, dirs['multispectral']),
                base_path_thermal=None,
                base_path_genotype=dirs['genotype'],
                base_path_dem=None,
                normalize_genotype=True,
                subsample_size=32,
                split_id=run['split'], # A single json containing all samples in test split, 
                #BUT, need to use splits for choosing correct normalizers...
                split_name='test',
                resized=True,
                return_dates=False,
                ms_standardize_path=dirs['standardize_path'], # Can be None
                genotype_scaler_path=dirs['genotype_scaler_path'],
                target_standardize_path=dirs['target_standardize_path'],
                )
    
    dataloader=DataLoader(ds,
                batch_size=16,
                collate_fn=variable_length_collate_fullfusedbags,
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
        print(f"Temporal encoding: {model.model.temporal_encoding}")
        print(f"Channel encoding: {model.model.channel_encoding}")
        # GPU
        if run['use_gpu']:
            # Transfer the model on the GPU
            model.cuda()

        # TODO: change options here
        model.model.thermal = False
        model.model.dem = False
        model.model.genotype = dirs['genotype'] is not None
        if run['only_genotype']:
            model.model.multispectral = False

        y_true, y_pred, results = eval_model_regression(model.model, dataloader, 
                    use_gpu=run['use_gpu'], only_genotype=run['only_genotype'])
        
        # Add predictions to df and save
        #df_new=pd.DataFrame([y_pred, y_true], columns=['PREDICTED_YIELD','ACTUAL_YIELD_(VALID)'])
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
