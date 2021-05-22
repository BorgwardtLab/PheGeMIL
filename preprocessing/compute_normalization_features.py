# ------------------------------------------
# Short script to compute and store normalizing
# features for all images in a given set.
# 
# 2020.11.08
# ------------------------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import json

BASE_PATH = '/links/groups/borgwardt/Data/Jesse_2018/'
DATA_PATH = os.path.join(BASE_PATH, 'numpy_MIL_resized')



# Utils functions

def get_split_from_config(csv_path, split_id, split_name):
    # Copied utility from the dataloader class
    json_file = os.path.splitext(csv_path)[0] + f'_splits.json'
    config_file = os.path.join(json_file)
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config[str(split_id)][split_name]

def compile_running_statistics(imgs_paths):
    # Compute running mean and standard deviation per channel.
    count_imgs = 0
    count_files = 0
    sum_means = np.zeros(5) # for each channel
    sum_squares = np.zeros(5)
    for i, fname in enumerate(tqdm(imgs_paths)):
        imgs = np.load(fname)
        # Compute mean
        channel_mean = imgs.mean(axis=(0,2,3))

        count_imgs += imgs.shape[0]
        count_files += 1

        # Add weighted mean
        sum_means += channel_mean*imgs.shape[0]

        # Add sum of squares
        sum_squares += np.sum(imgs**2, axis=(0,2,3))/(128**2)
    mean = sum_means / count_imgs
    std = np.sqrt(sum_squares/count_imgs - mean**2)
    return mean, std


def main():
    # Load the relevant fold
    csv_path = os.path.join(BASE_PATH, 'csv','df_20200121_numpy_MIL_npy_coordinates_dates_genofiltered_dems.csv')
    df = pd.read_csv(csv_path)
    results = []
    # Load the split files and only keep the relevant split
    for split in range(5):
        print(f'Computing mean and std for split {split}')
        idx = get_split_from_config(csv_path, split, 'train')
        df_interest = df.iloc[idx].reset_index() # Reset the index for later querrying
        
        # Iterate over all available images of the training set
        imgs_base_path = os.path.join(BASE_PATH, 'numpy_MIL_resized','2017-2018_CIMMYT_Wheat')
        imgs_paths = imgs_base_path + '/' + df_interest['Path'] + '/' + df_interest['Filename']

        # Get mean and std:
        mean_split, std_split = compile_running_statistics(imgs_paths)
        print("The means are:")
        print(mean_split)
        print("The std are:")
        print(std_split)
        print()

        results.append(np.hstack([mean_split, std_split]))
    
    # Combine and save results
    df_results = pd.DataFrame(results, columns = ['mean_r', 'mean_g', 'mean_b','mean_nir','mean_re', 
                'std_r', 'std_g', 'std_b','std_nir','std_re']).to_csv(os.path.join(BASE_PATH, 'csv', 'df_20201108_normalization_features.csv'))


    


if __name__ == "__main__":
    main()