# ------------------------------------------
# Short script to cut images in half for
# Training on single beds
# 2020.11.24
# ------------------------------------------
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from torch import from_numpy # Numpy like wrapper
from torchvision import transforms

from src.data import get_transformations, CropTransforms, transf_image

DATA_FOLDER = '/links/groups/borgwardt/Data/Jesse_2018'
input_data_path = os.path.join(DATA_FOLDER,'numpy_MIL_resized', '2017-2018_CIMMYT_Wheat')
processed_data_path = os.path.join(DATA_FOLDER, 'numpy_MIL_resized', 'half_cropped_ms_images') 
# Directly transform the images as well.

# Pass through all the images and cut them in half.
# First, list the different folders
folders = [f for f in os.listdir(input_data_path) if os.path.isdir(os.path.join(input_data_path, f))]
print(folders)

# Create destination folder
os.makedirs(processed_data_path, exist_ok=True)

# Start by loading the csv master file
df_all = pd.read_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20191014_numpy_MIL_npy_coordinates.csv'))
print('csv file loaded')

# Create output paths
for path in df_all['Path'].unique():
    o = os.path.join(processed_data_path, path)
    os.makedirs(o, exist_ok=True)

# Info that need to be added: Path, which will be a constant, since all images are grouped into one npy file.

# Go through all the PLOT IDs
np_filenames = []

# Prepare transformers
transforms = get_transformations(augmentation='none', size=(128,128), fixed_size=False)
transform = CropTransforms(transforms, normalize=False)
for idx, row  in tqdm(df_all.iterrows()):
    # Save other params
    imgs = []
    imgs_original = np.load(os.path.join(input_data_path, row['Path'], row['Filename']))
    for img in imgs_original:
        # Chop image in 2 (not very elegant)
        h1 = img[:,:,:64]
        h2 = img[:,:,64:]
        # Reshape, resize the image and save as numpy
        h1 = transform(transf_image(h1))
        h2 = transform(transf_image(h2))
        imgs.append(h1.numpy())
        imgs.append(h2.numpy())

    imgs = np.asarray(imgs)
    # Save imgs to npy archive and other parameters to new df
    output_filepath = os.path.join(processed_data_path, row['Path'], row['Filename'])
    np.save(output_filepath, imgs)
print('Done.')
