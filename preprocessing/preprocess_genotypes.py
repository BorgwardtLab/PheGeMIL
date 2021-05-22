# ------------------------------------------
# Short script to prepare a new dataframe with the relevant GIDs for original training
# 
# 2020.11.08
# ------------------------------------------
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

BASE_PATH = '/links/groups/borgwardt/Data/Jesse_2018/'
DATA_PATH = os.path.join(BASE_PATH, 'genotypes')
output_path = os.path.join(BASE_PATH, 'numpy_MIL_resized', 'genotypes')

# Start by loading the GIDs of interest
# The notebook 20201104_revised_exploration shows that all
# original GIDs are contained in the new genotypes
# gids = np.loadtxt(os.path.join(DATA_PATH, 'gids_2013_2020_full.txt', dtype=str)) # No need to do so, we will save all of them to file.
# original_gids = np.loadtxt(os.path.join(DATA_PATH, 'gids.txt', dtype=str)

# Load the snps
df_full = pd.read_csv(os.path.join(DATA_PATH, '20201028_df_genotypes_2013_2020_full.csv'))
print('Loaded.')
df_full.set_index('Unnamed: 0', inplace=True)
for i,row in tqdm(df_full.iterrows()):
    np.save(os.path.join(output_path, f'{i}.npy'), np.array(row))
