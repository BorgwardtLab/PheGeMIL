# ------------------------------------------
# Short script to transform the tiff images from the new test set
# into npy arrays for faster loading
# THIS SCRIPT IS FOR Or_crop IMAGES
# 2020.11.05: need to regroup the images according to their shared study ID.
# ------------------------------------------
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import rasterio  # need to run: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tomatteo/.local/lib in gpu07
from affine import Affine

from torch import from_numpy # Numpy like wrapper
from torchvision import transforms

from src.data import get_transformations, CropTransforms, transf_image

DATA_FOLDER = '/links/groups/borgwardt/Data/Jesse_2018/202010_new_testset'
raw_data_path = os.path.join(DATA_FOLDER, '2018_CIMMYT_EYT_Flat')
processed_data_path = os.path.join(DATA_FOLDER, 'numpy_MIL_resized', 'multispectral_images_Or_large_crop') 
# Directly transform the images as well.

# First, list the different folders
folders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]
print(folders)

# Create destination folder
os.makedirs(processed_data_path, exist_ok=True)


# Start by loading the csv master file
df_all = pd.read_excel(os.path.join(DATA_FOLDER, 'csv', 'originals', '18MX_EYTBW_F5I_GRYLD.xlsx'))
print('csv file loaded')
# Info that need to be added: Path, which will be a constant, since all images are grouped into one npy file.
# Dates,
# unique_plot_ids = df_all.plot_id.unique() # There is one row per plot_id in that file.
def load_npy_img(filepath):
    dataset = rasterio.open(filepath)
    # Get coordinates
    x,y = Affine.from_gdal(*dataset.transform)*(0,0)

    # Transform to numpy, only read the first 5 channels.
    np_img = dataset.read()[:5].astype(np.float32)

    return np_img, x, y

# Prepare transformers
transforms = get_transformations(augmentation='none', size=(128,128), fixed_size=False)
transform = CropTransforms(transforms)


coordinates_x = []
coordinates_y = []
np_filenames = []
dates_df = []
paths = []
# Iterate over unique plot ids:
for idx, row in tqdm(df_all.iterrows()):
    # Iterate over folder and check if file is present images
    dates_row = []
    imgs = []
    for folder in folders:
        # Get filenames
        folder_path = os.path.join(raw_data_path, folder, 'filtered')
        fnames = [filename for filename in os.listdir(folder_path) if filename.startswith(row['plot_id'])]
        for fname in fnames:
            filename = os.path.join(folder_path, fname)
            if os.path.isfile(filename):
                # load the image 
                img, x, y = load_npy_img(filename)
                img = transf_image(img) # reshape the rasterio input
                imgs.append(transform(img).numpy()) # change to 128x128
                dates_row.append(folder.split('_')[-1][4:])
        
    # Save to file
    imgs = np.asarray(imgs)
    output_filename = os.path.join(processed_data_path, row['plot_id']+'.npy')
    np.save(output_filename, imgs)

    # Add info for csv
    dates_df.append(dates_row)
    paths.append('')
    np_filenames.append(row['plot_id']+'.npy')
    coordinates_x.append(x)
    coordinates_y.append(y)

print('Images transformed. Updating master csv file.')
df_all['coordinates_x'] = coordinates_x
df_all['coordinates_y'] = coordinates_y
df_all['Filename'] = np_filenames
df_all['Path'] = paths
df_all['dates'] = dates_df

df_all.to_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20201105_numpy_MIL_npy_coordinates.csv'), index=False)