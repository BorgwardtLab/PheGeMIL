# Script to preprocess the Digital Elevation Model images received by Kevin

import numpy as np
import pandas as pd
import rasterio # need to run: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tomatteo/.local/lib in gpu07

import os

from tqdm import tqdm

BASE_PATH = '/links/groups/borgwardt/Data/Jesse_2018/'
raw_data_path = os.path.join(BASE_PATH, 'height_images')
processed_data_path = os.path.join(BASE_PATH, 'numpy_MIL', 'height_images')

# First, list the different folders
folders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]

# Create destination folder
os.makedirs(processed_data_path, exist_ok=True)

def load_npy_img(filepath):
    dataset = rasterio.open(filepath)
    # Only keep first occurence as all bands are the same
    np_img = dataset.read(1)
    # Get the value used for nodata
    no_data = dataset.meta['nodata']
    min_val = np_img[np_img>=0].min()
    np_img[np_img<0] = min_val
    np_img = np_img - min_val
    return np_img

# Initiate the dataframe for paths and information
df = []

# Iterate over folder and collect images
for folder in folders:
    print(f'Processing images in the {folder} folder...')
    # Get filenames
    folder_path = os.path.join(raw_data_path, folder)
    thermal_filenames = np.array([f for f in os.listdir(folder_path) if f.endswith('.tif')])

    # There are more dates per image, we need to save the dates for the DEM images too.

    # Create destination folder
    os.makedirs(os.path.join(processed_data_path, folder), exist_ok=True)
    
    # Group by plotID, the filename is composed as follows: '18-OBR-YTBW-B5I-<PLOTID>.tif'
    plot_ids = np.array([n.split('-')[4][:-4] for n in thermal_filenames])

    # Iterate through plot_ids
    for pid in tqdm(np.unique(plot_ids)):
        # CAREFUL: only folder for which the script failed.
        if folder == '18MX_YT_1-60_Rededge_20180319_dem_crop':
            imgs_paths = thermal_filenames[plot_ids == pid]
            npy_img = np.stack([load_npy_img(os.path.join(folder_path, img)) for img in imgs_paths])
            np.save(os.path.join(processed_data_path, folder, f'{pid}.npy'), npy_img)
        else:
            npy_img = np.load(os.path.join(processed_data_path, folder, f'{pid}.npy'))
        df.append([pid, folder, npy_img.shape])
    print('Done.')
    print()
    # A second script will be used to group all images ACROSS dates and resize them.

# Save summary information to csv.
pd.DataFrame(df, columns=['PlotID', 'Path', 'Shape']).to_csv(
    '/links/groups/borgwardt/Data/Jesse_2018/csv/df_20200119_numpy_demimg.csv',
    index=False)





