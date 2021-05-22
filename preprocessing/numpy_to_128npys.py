# ------------------------------------------
# Short script to save a single npy file for 
# each plotID and preprocess (resizing) the images to 
# speed up loading time.
# ------------------------------------------
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from torch import from_numpy # Numpy like wrapper
from torchvision import transforms

from src.data import get_transformations, CropTransforms, transf_image


DATA_FOLDER = '/links/groups/borgwardt/Data/Jesse_2018/'
# Start by loading the csv master file
df_all = pd.read_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20190717_numpy_coordinates.csv'))

# Get rid of uninteresting columns
df_all.drop(columns=['Unnamed: 0','iyear', 'ilocation',
       'itrial', 'icondition', 'site', 'year', 'location',
       'cycle', 'conditions', 'occ'], inplace=True)

# Set up paths
input_path = os.path.join(DATA_FOLDER, 'numpy', '2017-2018_CIMMYT_Wheat')
output_path = os.path.join(DATA_FOLDER, 'numpy_MIL_resized', '2017-2018_CIMMYT_Wheat')

# Create output paths
for path in df_all['Path'].unique():
    o = os.path.join(output_path, path)
    os.makedirs(o, exist_ok=True)

# Go through all the PLOT IDs
np_filenames = []
input_img_paths = input_path + '/' + df_all['Path'] + '/' + df_all['Filename']
# Start by grouping the df by plot ID.
bags_indices_list = list(df_all.groupby('PlotID').indices.values())
new_df = []

# Prepare transformers
transforms = get_transformations(augmentation='none', size=(128,128), fixed_size=False)
transform = CropTransforms(transforms)
for idx_list in tqdm(bags_indices_list):
    # Save other params
    imgs = []
    fnames = []
    for idx in idx_list:
        # Load npy image
        img = np.load(input_img_paths[idx])
        img = transf_image(img)
        # Resize the image and save as numpy
        img = transform(img)
        imgs.append(img.numpy())
        fnames.append(df_all['Filename'][idx])

    imgs = np.asarray(imgs)
    # Save imgs to npy archive and other parameters to new df
    row = df_all.iloc[idx_list[0]]
    output_filepath = os.path.join(output_path, row['Path'], row['PlotID']+'.npy')
    np.save(output_filepath, imgs)
    old_columns = ['Path', 'PlotID', 'trial','rep','subblock','col','row','entry','gid','tid','DAYSMT',
                    'DTHD',	'GERMPCT', 'GRYLD',	'NOTES', 'PH', 'coordinates_x', 'coordinates_y']
    new_row = [row[i] for i in old_columns] + [row['PlotID']+'.npy'] + [';'.join(fnames)]
    new_df.append(new_row)

# Save new df in csv.
print("New MIL numpy arrays saved. Saving CSV...")
df_new = pd.DataFrame(new_df, columns=old_columns+['Filename']+['Old_filenames'])
df_new.to_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20191014_numpy_MIL_npy_coordinates.csv'), index=False)

