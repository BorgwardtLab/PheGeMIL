# ------------------------------------------
# Script to extract VI measurements from numpy images.
# ------------------------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

DATA_FOLDER = '/links/groups/borgwardt/Data/Jesse_2018/'
# Start by loading the csv master file
df_all = pd.read_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20190717_numpy_coordinates.csv'))

# Set up paths
images_path = os.path.join(DATA_FOLDER, 'numpy', '2017-2018_CIMMYT_Wheat')

# Vegetation indeces:
def ndvi(img):
    return np.divide(img[:,:,3]-img[:,:,0], img[:,:,3]+img[:,:,0])

def gndvi(img):
    return np.divide(img[:,:,3]-img[:,:,1], img[:,:,3]+img[:,:,1])

def rendvi(img):
    return np.divide(img[:,:,3]-img[:,:,4], img[:,:,3]+img[:,:,4])

def endvi(img):
    return np.divide(img[:,:,3]+img[:,:,1]-2*img[:,:,2], img[:,:,3]+img[:,:,1]+2*img[:,:,2])

def gipvi(img):
    return np.divide(img[:,:,3],img[:,:,3]+img[:,:,1])

def sumstats(single_channel_img):
    # Add min, max, mean, std 
    return np.min(single_channel_img), np.max(single_channel_img), \
        np.mean(single_channel_img), np.std(single_channel_img)

# Go through all the images
sumstats_dict = {'ndvi': [], 'gndvi': [], 'rendvi': [], 'endvi': [], 'gipvi': []}
vi_names = ['ndvi', 'gndvi', 'rendvi', 'endvi', 'gipvi']
for idx,row in tqdm(df_all.iterrows()):
    # Load image
    filepath = os.path.join(images_path, row['Path'], row['Filename'])
    img = np.load(filepath)

    for i, f in enumerate([ndvi, gndvi, rendvi, endvi, gipvi]):
        sumstats_dict[vi_names[i]].append(sumstats(f(img)))

# Add to dataframe
for vi in vi_names:
    sumstats_dict[vi] = np.asarray(sumstats_dict[vi])
    for i, sup in enumerate(['min', 'max', 'mean', 'std']):
        df_all[vi+'_'+sup] = sumstats_dict[vi][:,i]


df_all.to_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20190718_numpy_coordinates_VI.csv'))