# ------------------------------------------
# Short script to save a single npy file for 
# each thermal image plotID and preprocess 
# (resizing) the images to 
# speed up loading time.
# ------------------------------------------
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from torch import from_numpy # Numpy like wrapper
from torchvision import transforms

from src.data import get_transformations, CropTransforms, transf_image, uint_to_float


DATA_FOLDER = '/links/groups/borgwardt/Data/Jesse_2018/'
# Start by loading the csv master file
df_all = pd.read_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20191031_numpy_thermalimg.csv'))

# Set up paths
input_path = os.path.join(DATA_FOLDER, 'numpy_MIL', 'thermal_images')
output_path = os.path.join(DATA_FOLDER, 'numpy_MIL_resized', 'thermal_images')

# No need for output paths as we will have one npy file per plot ID
# (There are at most 2 dates per plot)

# Go through all the PLOT IDs
np_filenames = []
input_img_paths = input_path + '/' + df_all['Path'] + '/' + df_all['PlotID'].apply(str) + '.npy'
# Start by grouping the df by plot ID.
bags_indices_list = list(df_all.groupby('PlotID').indices.values())
new_df = []

# Prepare transformers #IMPORTANT TO CHOOSE THE RIGHT RESIZING
transforms = get_transformations(augmentation='none', size=(40,40), fixed_size=False,
                reshaping='resize')
# No need to use the Class as we have 1 channel only
# transform = CropTransforms(transforms)
for idx_list in tqdm(bags_indices_list):
    # Save other params
    imgs_bag = []
    paths = []
    shapes = []
    for idx in idx_list:
        # Load npy images (they're already in bags per date)
        imgs = np.load(input_img_paths[idx])
        imgs = uint_to_float(imgs, bits=8)
        # Resize the image and save as numpy
        imgs_bag += [transforms(img).numpy() for img in imgs]
        paths.append(df_all['Path'][idx])

    imgs = np.asarray(imgs_bag)
    # Save imgs to npy archive and other parameters to new df
    row = df_all.iloc[idx_list[0]]
    output_filepath = os.path.join(output_path, str(row['PlotID'])+'.npy')
    np.save(output_filepath, imgs)

    new_row = [row['PlotID'], ';'.join(paths), imgs.shape]
    new_df.append(new_row)

# Save new df in csv.
print("New MIL numpy arrays saved. Saving CSV...")
df_new = pd.DataFrame(new_df, columns=['PlotID', 'old_paths', 'shape'])
df_new.to_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20191119_numpy_thermalimg_resized.csv'), index=False)
