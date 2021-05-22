# Script to preprocess the thermal images received by Kevin

import numpy as np
import pandas as pd
import rasterio # need to run: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tomatteo/.local/lib in gpu07

import os

from tqdm import tqdm
from PIL import Image

BASE_PATH = '/links/groups/borgwardt/Data/Jesse_2018/'
raw_data_path = os.path.join(BASE_PATH, 'thermal_images')
processed_data_path = os.path.join(BASE_PATH, 'numpy_MIL', 'thermal_images')

# First, list the different folders
folders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]

# Create destination folder
os.makedirs(processed_data_path, exist_ok=True)

def load_npy_img(filepath):
    # dataset = rasterio.open(filepath)
    # Only keep first occurence as all bands are the same
    # return dataset.read(1)
    # Since we use tiff images and nothing about their geo component, we can load them with Image directly
    pil_img = Image.open(filepath)
    np_img = np.array(pil_img.getdata())[:,0]
    w, h = pil_img.size
    np_img.shape = (h, w)
    return np_img

# Initiate the dataframe for paths and information
df = []

# Iterate over folder and collect images
for folder in folders:
    print(f'Processing images in the {folder} folder...')
    # Get filenames
    folder_path = os.path.join(raw_data_path, folder)
    thermal_filenames = np.array(os.listdir(folder_path))

    # Create destination folder
    os.makedirs(os.path.join(processed_data_path, folder), exist_ok=True)
    
    # Group by plotID, the filename is composed as follows: '18-OBR-YTBW-B5I-<PLOT_ID>-<DATE>_<TIME>.tif'
    plot_ids = np.array([n.split('-')[4] for n in thermal_filenames])

    # Iterate through plot_ids
    for pid in tqdm(np.unique(plot_ids)):
        imgs_paths = thermal_filenames[plot_ids == pid]
        npy_img = np.stack([load_npy_img(os.path.join(folder_path, img)) for img in imgs_paths])

        np.save(os.path.join(processed_data_path, folder, f'{pid}.npy'), npy_img)
        df.append([pid, folder, npy_img.shape])
    print('Done.')
    print()

# Save summary information to csv.
pd.DataFrame(df, columns=['PlotID', 'Path', 'Shape']).to_csv(
    '/links/groups/borgwardt/Data/Jesse_2018/csv/df_20191031_numpy_thermalimg.csv',
    index=False)





