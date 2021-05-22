# ------------------------------------------
# Short script to transform the tiff images 
# into npy arrays for faster loading
# ------------------------------------------
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import rasterio  # need to run: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tomatteo/.local/lib in gpu07
from affine import Affine

DATA_FOLDER = '/links/groups/borgwardt/Data/Jesse_2018/'
# Start by loading the csv master file
df_all = pd.read_csv(os.path.join(DATA_FOLDER, 'csv', 'clean_df_20190716.csv'))

# Set up paths
images_path = os.path.join(DATA_FOLDER, 'uncompressed', '2017-2018_CIMMYT_Wheat')
output_path = os.path.join(DATA_FOLDER, 'numpy', '2017-2018_CIMMYT_Wheat')

# Create output paths
for path in df_all['Path'].unique():
    o = os.path.join(output_path, path)
    os.makedirs(o, exist_ok=True)

# Go through all the images
coordinates_x = []
coordinates_y = []
np_filenames = []
for idx,row in tqdm(df_all.iterrows()):
    # Load image
    filepath = os.path.join(images_path, row['Path'], row['Filename'])
    dataset = rasterio.open(filepath)

    # Get coordinates
    x,y = Affine.from_gdal(*dataset.transform)*(0,0)
    coordinates_x.append(x)
    coordinates_y.append(y)

    # Transform to numpy
    np_img = dataset.read().astype(np.float32)

    # Save to new file and add columns
    pre, ext = os.path.splitext(row['Filename'])
    new_filepath = pre + '.npy'
    output_filepath = os.path.join(output_path, row['Path'], new_filepath)
    np.save(output_filepath, np_img)
    np_filenames.append(new_filepath)

print('Images transformed. Updating master csv file.')
df_all['coordinates_x'] = coordinates_x
df_all['coordinates_y'] = coordinates_y
df_all['Filename'] = np_filenames

df_all.to_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20190717_numpy_coordinates.csv'), index=False)
