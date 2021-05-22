import numpy as np
import pandas as pd
import torch
import os
import csv
from tqdm import tqdm
import pickle
from types import MethodType
from scipy.stats import pearsonr



from .data import *
from .modules import *
from .main_geno import *

# Workaround for resource FD freeing
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def get_saved_checkpoint(folder):
    # Manual extraction of the epoch number for the
    for f in os.listdir(folder):
        if os.path.splitext(f)[1] == '.ckpt':
            return f

# We need to modify the forward method to allow for empty sets
def forward(self, x):
    # First unroll the input
    x_multispectral, lengths_multispectral, x_thermal, lengths_thermal, \
        x_genotype, dates_multispectral, dates_thermal  = self._unroll_input(x)
    # TODO: to save on computation, avoid encoding empty images.
    if x_multispectral.size()[1]>0:
        x_multispectral = self.multispectral_encoder(x_multispectral)
        x_multispectral = self.pad_0(self.pad_0(self.pad_1(x_multispectral)))
    # if x_thermal.size()[1]>0:
    #     x_thermal = self.thermal_encoder(x_thermal)
    #     x_thermal = self.pad_0(self.pad_1(self.pad_0(x_thermal)))
    if self.genotype:
        x_genotype = self.genotype_encoder(x_genotype)
        x_genotype = self.pad_1(self.pad_0(self.pad_0(x_genotype)))
        
    x = x_genotype.unsqueeze(1)
    if x_multispectral.size()[1]>0:
        x = torch.cat([x_multispectral, x], dim=1)
    # if x_thermal.size()[1]>0:
    #     x = torch.cat([x_thermal, x], dim=1)
    lengths = [lengths_multispectral, lengths_thermal]
    
    # Aggregate
    x = self.aggregator(x, lengths)

    batch_size, n_heads, L = x.size()
    x = x.view(batch_size, L*n_heads)
    return self.regressor(x)

def filter_images(imgs, dates, latest_time):
    # Filter out images in the future
    bool_filter = dates[:,:,:latest_time].sum(dim=2)
    leng = bool_filter.sum().type(torch.long)
    return imgs[bool_filter.type(torch.bool)].unsqueeze(0), leng.unsqueeze(0)

def test_at_time(model, dataloader, latest_time=15):
    if latest_time > 15:
        print("Latest time cannot be > 15")
        return
    
    y_preds = []
    all_labels = []
        
    for batch in tqdm(dataloader):
        data_multispectral, data_thermal, data_genotype, labels, lengths_multispectral, \
                lengths_thermal, dates_multispectral, dates_thermal = batch
        # Filter out images in the future
        data_multispectral, lengths_multispectral = filter_images(data_multispectral, dates_multispectral, latest_time)
        data_thermal, lengths_thermal = filter_images(data_thermal, dates_thermal, latest_time)
        y_pred = model((data_multispectral, lengths_multispectral, data_thermal,
                               lengths_thermal, data_genotype)).cpu().detach().numpy()
        y_preds.append(y_pred)
        all_labels.append(labels.numpy())
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

    loss = model.loss(
        torch.tensor(all_labels), torch.tensor(y_preds)).item()
    # Compute other metrics MSE, r2, pearson
    mae = torch.mean(torch.abs(torch.tensor(all_labels)-torch.tensor(y_preds))).numpy()
    r2 = r2_score(all_labels, y_preds)
    pearson_score = pearsonr(all_labels, y_preds)[0]
    
    return loss, mae, r2, pearson_score

def main():
    # Need a test set that returns dates
    transforms = None
    size = (128,128) # if self.hparams.encoder_type == 'convnet' else (224,224)
    # Add path information
    fixed_size = True
    transforms = get_transformations(augmentation='none', size=size, fixed_size=fixed_size)
    tmp_folder = 'matteos_cached_resized_imgs'

    base_path_multispectral = os.path.join('/tmp', tmp_folder, '2017-2018_CIMMYT_Wheat')
    base_path_thermal = os.path.join('/tmp', tmp_folder, 'thermal_images')
    base_path_genotype = os.path.join('/tmp', tmp_folder, 'genotypes')

    results = {}
    for fold_id in range(5):
        results[fold_id] = {}
        print(f"Computing results for model {fold_id}")
        folder = f'/links/groups/borgwardt/Data/Jesse_2018/output/torch_exp_logs/exp_logs/temporal_thermal_best_params_cv_with_geno/version_{fold_id}'
        model1 = MILCropYieldGeno.load_from_metrics(
                        weights_path=os.path.join(folder,get_saved_checkpoint(folder)),
                        tags_csv=os.path.join(folder, 'meta_tags.csv'),
                        on_gpu=True)
        # Modify model:
        model1.model.forward = MethodType(forward, model1.model)
        model1.eval()
        model1.freeze()
        # Latest_time is the latest_date for which we want to keep information
        test_dataset = FusedGenoBags(os.path.join('/links/groups/borgwardt/Data/Jesse_2018','csv', 
                                            'df_20200107_numpy_MIL_npy_coordinates_dates_genofiltered.csv'),
                    base_path_multispectral=base_path_multispectral,
                    base_path_thermal=base_path_thermal,
                    base_path_genotype=base_path_genotype,
                    normalize_genotype=True,
                    subsample_size=None,
                    split_id=fold_id,
                    split_name='test',
                    resized=fixed_size,
                    return_dates=True
                    )
        test_loader = DataLoader(
                dataset=test_dataset,
                collate_fn=variable_length_collate_fusedbags,
                batch_size=1, # Small batch_size to avoid memory overflow for large bags
                shuffle=False,
                # When using caching we can onlyu use a single worker
                # num_workers=0 if self.hparams.cache_dataset else 8, # 1 for CUDA?
                num_workers=12, # 1 for CUDA?
            )
        print('Data loaded.')
        for t in range(1,17):
            results[fold_id][t] = {}
            loss, mae, r2, pearson_score = test_at_time(model1, test_loader, latest_time=t)
            results[fold_id][t]['mse'] = loss
            results[fold_id][t]['mae'] = mae
            results[fold_id][t]['r2'] = r2
            results[fold_id][t]['pearson'] = pearson_score
            print(f'\tResults for t {t}: MSE: {loss}, MAE: {mae}, R2: {r2}, Pearson: {pearson_score}')

    # Save to pickle
    with open("/links/groups/borgwardt/Data/Jesse_2018/output/torch_exp_logs/exp_logs/20200117_ablation_results_ms_only.pkl","wb") as f:
        pickle.dump(results,f)
    
if __name__ == "__main__":
    main()