# ------------------------------------------
# Dataset classes and loading functions
# ------------------------------------------

import pandas as pd
import numpy as np
import os
import random

import torch
from torch.utils.data.dataset import Dataset
from torch import from_numpy # Numpy like wrapper
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

import json
from torch import tensor
import pickle as pkl

class CropBagDataset(Dataset):
    """Dataset wrapping images BAGS and target labels for the plot aerial images,
        based on the CSV master file provided by Poland.
    Arguments:
        A CSV file path
        A base path where the files are located
        A set of pytorch transformation
        (Optional) a maximum number of samples sampled per bag
        (Optional) an indicator whether to return encoded
    """
    def __init__(self, csv_path, base_path, transform=None, subsample_size=None,
                split_name='train', split_id=0, verbose=True, resized=True,
                return_dates=False, standardize_path=None, standardize_target_path=None):
        self.df = pd.read_csv(csv_path)
        self.split_id = split_id
        self.split_name = split_name
        self.csv_path = csv_path

        self.standardize = False
        self.standardize_target = False
        self.resized = resized
        self.return_dates = return_dates # NOTE: only tested with npyformat='.npy' and resized=True
        if self.return_dates:
            # If requires dates ensure they are present
            assert 'dates_enc' in self.df.columns
            # need to convert back to int
            self.df.dates_enc = self.df.dates_enc.apply(lambda x: [int(i) for i in x[1:-1].split()])
       
        # Load standardization values for resized image
        if standardize_path is not None and resized:
            self.standardize = True
            df_standardized = pd.read_csv(standardize_path, index_col=0)
            self.training_mean = df_standardized.iloc[self.split_id][[f'mean_{c}' for c in ['r', 'g', 'b', 'nir', 're']]].values
            self.training_std = df_standardized.iloc[self.split_id][[f'std_{c}' for c in ['r', 'g', 'b', 'nir', 're']]].values


        # Load standardization values for target
        if standardize_target_path is not None:
            self.standardize_target = True
            df_standardized = pd.read_csv(standardize_target_path, index_col=0)
            self.target_mean = df_standardized.iloc[self.split_id]['mean_yield']
            self.target_std = df_standardized.iloc[self.split_id]['std_yield']

        # Load the split files and only keep the relevant split
        idx = self._get_split_from_config()
        self.df = self.df.iloc[idx]
        # Reset the index for later querrying
        self.df.reset_index(inplace=True)
        if verbose:
            print(f'There are {len(self.df)} entries for {split_name} of the {split_id}th split.')

        self.base_path = base_path

        self.transform = transform
        self.npyformat = os.path.splitext(self.df['Filename'].tolist()[0])[1]

        # Sometimes the bag size is quite large (up to 82 images)
        # Hence, we need to add random sampling
        self.subsample_size = subsample_size # if set to None, won't be subsampled

        # Generate bags
        if self.npyformat == '.npy' and not self.resized:
            self.bags_indices_list, self.labels_list = self._form_bags()
        else:
            self.labels_list = list(self.df['GRYLD'].astype(np.float32))

        self.X = base_path + '/' + self.df['Path'] + '/' + self.df['Filename']
        # print(self.X[~self.X.apply(lambda x: os.path.isfile(x))])
        assert self.X.apply(lambda x: os.path.isfile(x)).all(), "Some images referenced in the CSV file were not found"

    def _to_one_hot(self, x, max_len=16): # This param is already fixed by the dataset, but could be tunable
        # returns a one hot 
        one_hot = torch.zeros(len(x), max_len)
        one_hot[torch.arange(len(x)), x] = 1
        return one_hot

    def _form_bags(self):
        # Start by grouping the df by plot ID.
        bags_indices_list = list(self.df.groupby('PlotID').indices.values())

        # Manual verification showed that there is always only 1 target value
        labels_list = []
        for idx in bags_indices_list:
            labels_list.append(self.df.iloc[idx[0]]['GRYLD'].astype(np.float32))
        return bags_indices_list, labels_list
    
    def _get_split_from_config(self):
        json_file = os.path.splitext(self.csv_path)[0] + f'_splits.json'
        config_file = os.path.join(json_file)
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config[str(self.split_id)][self.split_name]

    def X(self):
        return self.X

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = []
        if self.npyformat == '.npy':
            if self.resized: # Only one used
                # This is if the images were already resized beforehand and stored in multidimensional 
                # numpy tensors
                imgs = np.load(self.X[index])
                idx_list = np.arange(len(imgs))
                if self.return_dates:
                    # return a one-hot encoded vector
                    dates = self._to_one_hot(self.df.loc[index,'dates_enc'])
                if self.subsample_size is not None:
                    if len(idx_list) > self.subsample_size:
                        idx_list = np.random.choice(idx_list, self.subsample_size, replace=False)
                        imgs = imgs[idx_list, :, :, :]
                        if self.return_dates:
                            dates = dates[idx_list,:]
                # NO TRANSFORMATIONS ALLOWED... TODO: implement later
                if self.standardize:
                    # torch transforms require special reordering of the channels, stick to numpy
                    imgs = np.moveaxis((np.moveaxis(imgs, 1, -1) - self.training_mean)/self.training_std,-1,1)
                    
                bag = imgs
                label = from_numpy(np.array(self.labels_list[index]).reshape(-1).astype(np.float32))
                if self.standardize_target:
                    label = (label - self.target_mean)/self.target_std
            else:
                idx_list = self.bags_indices_list[index]
                # Sometimes the bag size is quite large (up to 82 images)
                # Hence, we need to add random sampling
                if self.subsample_size is not None:
                    if len(idx_list) > self.subsample_size:
                        idx_list = np.random.choice(idx_list, self.subsample_size, replace=False)

                for idx in idx_list:
                    img = np.load(self.X[idx])
                    img = transf_image(img)
                
                    # TODO FOR AUGMENTATION
                    # TODO: change order of transformation to tensor
                    if self.transform is not None:
                        img = self.transform(img)
                    bag.append(img)
                label = from_numpy(self.labels_list[index].reshape(-1))
                bag = np.stack(bag)
        elif self.npyformat == '.npz':
            # If you want to experiment with reshapes, use this one (df_20190814_numpy_MIL_coordinates.csv)
            imgs = list(np.load(self.X[index]).values())
            idx_list = np.arange(len(imgs))
            if self.subsample_size is not None:
                if len(idx_list) > self.subsample_size:
                    idx_list = np.random.choice(idx_list, self.subsample_size, replace=False)
            for idx in idx_list:
                img = imgs[idx]
                img = transf_image(img)
                if self.transform is not None:
                    img = self.transform(img)
                bag.append(img)
            label = from_numpy(np.array(self.labels_list[index]).reshape(-1).astype(np.float32))
            bag = np.stack(bag)
        if self.return_dates:
            return bag, label, dates
        else:
            return bag, label

    def getDF(self):
        return self.df

###################################
# TRANSFORMATIONS
###################################
class CropTransforms(object):
    """The transforms class will normalize the images, take a transforms.Compose 
        object and apply it separately to the RGB image and to channels 4 and 5.
        Args:
            transform (transforms.Compose object): transforms operations to be performed, must return a Tensor object
            normalize (bool or list, default: True): if True or list, calls normalize_array with max_values
            img_type (str, default: multispectral): if multispectral use 16bits transforms and 5 channels, if thermal 8 bits and 1 channel
    """

    def __init__(self, transform, normalize=True, img_type='multispectral'):
        self.transform = transform
        self.normalize = normalize
        assert img_type in ['multispectral', 'thermal']
        self.img_type = img_type

    def __call__(self, img):
        """
        Args:
            img (Numpy array): Numpy array of size (H, W, 5) to be normalized and transformed.

        Returns:
            Tensor: Transformed Tensor image.
        """
        # TEST DIMENSIONS
        if self.img_type == 'multispectral':
            assert img.shape[-1] == 5
            # if necessary, normalize
            if self.normalize:
                if isinstance(self.normalize, list) and len(self.normalize):
                    img = normalize_array(img, max_values=self.normalize)
                else:
                    # img = normalize_array(img)
                    img = uint_to_float(img, bits=14)
                    
            
            # ISSUE: RANDOM SEED WILL MAKE OTHER TRANSFORMATIONS FOR EACH CHANNEL (CAN ONLY WORK WITH ONE WORKER!)
            seed = np.random.randint(42) # make a seed with numpy generator 
            random.seed(seed) # apply this seed to img tranfsorms
            # Transform independently
            imgs_rgb = (255.0 * img[:,:,:3]).astype(np.uint8)
            imgs_rgb = self.transform(imgs_rgb) 
            # Need to transform imgs_rgb to [0, 255] values for PIL
            random.seed(seed) # apply this seed to channel 4 tranfsorms
            imgs_4 = self.transform(np.expand_dims(img[:,:,3], axis=2))
            random.seed(seed) # apply this seed to channel 5 tranfsorms
            imgs_5 = self.transform(np.expand_dims(img[:,:,4], axis=2))
            
            if imgs_4.dim() < imgs_rgb.dim():
                print(hey)
                print(imgs_4.size())
                imgs_4 = imgs_4.unsqueeze(-1)
                imgs_5 = imgs_5.unsqueeze(-1)
                print(imgs_4.size())
            
            # Combine
            img_comb = torch.cat([imgs_rgb, imgs_4, imgs_5],0)
            return img_comb
        elif self.img_type == 'thermal':
            if self.normalize:
                img = uint_to_float(img, bits=16)
                return self.transform(img)



    def __repr__(self):
        return self.__class__.__name__ + " inner transformations:\n" + str(self.transform)

def get_transformations(augmentation='none', size=(225,225), fixed_size=False, reshaping='resize'):
    # Return the transformations array, for validation: no transformations
    # Options: 'none', 'light', 'strong'
    # reshaping can be either 'resize' for an interpolated resizing or 'pad' for a resize with padding
    # AVOID 'pad' and 'strong' together.
    base_transformations = []
    if augmentation in ['light', 'strong']:
        base_transformations += [transforms.ToPILImage()]
    if not fixed_size:
        # experiment with reshape here
        if reshaping =='resize':
            base_transformations = [transforms.ToPILImage(), transforms.Resize(size)]
        elif reshaping == 'pad':
            base_transformations = [transforms.ToPILImage(), transforms.CenterCrop(size)]
        else:
            raise Warning(f'The reshaping option should be either resize or pad (not {reshaping})')

    if augmentation == 'none':
        augm_transformations = []
    elif augmentation == 'light':
        augm_transformations = [transforms.RandomHorizontalFlip(),\
                                transforms.RandomVerticalFlip()]
    elif augmentation == 'strong':
        augm_transformations = [transforms.RandomResizedCrop(size[0], scale=(0.25,1.0)),\
                                transforms.ColorJitter(),\
                                transforms.RandomHorizontalFlip(),\
                                transforms.RandomVerticalFlip()]
    else:
        raise ValueError("{} is not a valid augmentation, please use 'none', 'light' or 'strong.".format(augmentation))

    return transforms.Compose(base_transformations + augm_transformations + [transforms.ToTensor()])#\
                                # [transforms.ToTensor(),
                                #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #                        std=[0.229, 0.224, 0.225])])

def get8(m):
    return int(m%255-np.floor(m/255))

def get255_equivalent(m):
    return m+255-get8(m)

def normalize_array(num_array, max_values=None):
    # Function that normalizes the values of the numpy image.
    # Can also accept a set of global max_values for normalization
    norm_array = np.zeros_like(num_array).astype(np.float32)
    # RGB norm:
    if max_values is not None:
        norm_array[:,:,:3] = num_array[:,:,:3] / get255_equivalent(max_values[0]) 
        # Channel 4 and 5
        norm_array[:,:,3] = num_array[:,:,3] / get255_equivalent(max_values[1])
        norm_array[:,:,4] = num_array[:,:,4] / get255_equivalent(max_values[2])
    else:
        norm_array[:,:,:3] = num_array[:,:,:3] / get255_equivalent(num_array[:,:,:3].max()) 
        # Channel 4 and 5
        norm_array[:,:,3] = num_array[:,:,3] / get255_equivalent(num_array[:,:,3].max())
        norm_array[:,:,4] = num_array[:,:,4] / get255_equivalent(num_array[:,:,4].max())
    return norm_array

# Images are aquired on a 14bit camera, need to be rescaled:
def uint_to_float(img, bits=16):
    return np.divide(img, 2**bits - 1).astype(np.float32)

# Function to move channel axis to end (C x H x W) -> (H x W x C)
def transf_image(np_img):
    return np.moveaxis(np_img, 0, -1)

def variable_length_collate(batch):
    """Combine data with variable length using padding.
    Further add a length tensor to the output, such that the invalid values can
    be appropriately masked.
    Args:
        batch: Nested list of instances
    Returns:
        data, labels, lengths tensors
    """

    imgs_bags = []
    labels = []
    dates_bag = []

    for batch_element in batch:
        # Check if dates are also returned or only imgs and label
        # We cannot do this with an argument
        if len(batch_element) == 2:
            return_dates = False
            imgs, label = batch_element
        elif len(batch_element) == 3:
            return_dates = True
            imgs, label, dates = batch_element
            dates_bag.append(dates)
        imgs_bags.append(imgs)
        labels.append(label)

    lengths = list(map(len, imgs_bags))
    max_len = max(lengths)

    extended_imgs_bags = []
    for imgs, length in zip(imgs_bags, lengths):
        padding = max_len - length
        extended_imgs_bags.append(
            np.pad(
                imgs, ((0, padding), (0, 0), (0, 0), (0, 0)),
                mode='constant'
            )
        )
    if return_dates: 
        extended_dates_bag = []
        for dates, length in zip(dates_bag, lengths):
            padding = max_len - length
            extended_dates_bag.append(
                np.pad(
                    dates, ((0, padding), (0, 0)),
                    mode='constant'
                )
            )
        return tensor(extended_imgs_bags, dtype=torch.float), \
            tensor(labels, dtype=torch.float),             \
            tensor(lengths, dtype=torch.long),             \
            tensor(extended_dates_bag, dtype=torch.float)
    else:
        return tensor(extended_imgs_bags, dtype=torch.float), \
            tensor(labels, dtype=torch.float),             \
            tensor(lengths, dtype=torch.long)

###################################
# Fusion dataset
###################################

class ThermalBags(Dataset):
    """Dataset wrapping images BAGS and target labels for the plot aerial images,
        based on the CSV master file provided by Poland.
    Arguments:
        A CSV file path
        A base path where the files are located
        A set of pytorch transformation
        (Optional) a maximum number of samples sampled per bag
    """
    def __init__(self, csv_path, base_path, transform=None, subsample_size=None,
                split_name='train', split_id=0, return_dates=False, date_col_name=None):
        self.df = pd.read_csv(csv_path)
        self.split_id = split_id
        self.split_name = split_name
        self.csv_path = csv_path

        self.return_dates = return_dates # NOTE: only tested with npyformat='.npy' and resized=True
        if self.return_dates:
            if date_col_name is not None:
                self.date_col_name = date_col_name
            else:
                self.date_col_name = 'dates_thermal_enc'
            # If requires dates ensure they are present, we use the same df as for the ms images
            assert self.date_col_name in self.df.columns
            # need to convert back to int
            self.df[self.date_col_name] = self.df[self.date_col_name].apply(lambda x: [int(i) for i in x[1:-1].split()])

        # Load the split files and only keep the relevant split
        idx = self._get_split_from_config()
        self.df = self.df.iloc[idx]
        # Reset the index for later querrying
        self.df.reset_index(inplace=True)

        self.base_path = base_path

        self.transform = transform

        self.labels_list = list(self.df['GRYLD'].astype(np.float32))

        # Sometimes the bag size is quite large (up to 82 images)
        # Hence, we need to add random sampling
        self.subsample_size = subsample_size # if set to None, won't be subsampled

        self.X = base_path + '/' + self.df['PlotID'].apply(lambda x: x.split('-')[-1]) + '.npy'
        assert self.X.apply(lambda x: os.path.isfile(x)).all(), "Some THERMAL images referenced in the CSV file were not found"
    
    def _to_one_hot(self, x, max_len=16): # This param is already fixed by the dataset, but could be tunable
        # returns a one hot 
        one_hot = torch.zeros(len(x), max_len)
        one_hot[torch.arange(len(x)), x] = 1
        return one_hot

    def _get_split_from_config(self):
        json_file = os.path.splitext(self.csv_path)[0] + f'_splits.json'
        config_file = os.path.join(json_file)
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config[str(self.split_id)][self.split_name]

    def X(self):
        return self.X

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = []
        # This is if the images were already resized beforehand and stored in multidimensional 
        # numpy tensors
        imgs = np.load(self.X[index])
        idx_list = np.arange(len(imgs))
        if self.return_dates:
            # return a one-hot encoded vector
            dates = self._to_one_hot(self.df.loc[index,'dates_thermal_enc'])
        if self.subsample_size is not None:
            if len(idx_list) > self.subsample_size:
                idx_list = np.random.choice(idx_list, self.subsample_size, replace=False)
                imgs = imgs[idx_list, :, :, :]
                if self.return_dates:
                    dates = dates[idx_list,:]
        # NO TRANSFORMATIONS ALLOWED... TODO: implement later
        bag = imgs
        label = from_numpy(np.array(self.labels_list[index]).reshape(-1).astype(np.float32))
        if self.return_dates:
            return bag, label, dates
        else:
            return bag, label

    def getDF(self):
        return self.df



class FusedBags(Dataset):
    """Dataset wrapping multispectral and thermal images.
    Arguments:
        A CSV file path
        A base path where the files are located
        A set of pytorch transformation
        (Optional) a maximum number of samples sampled per bag
    """
    def __init__(self, csv_path, base_path_multispectral, base_path_thermal,
                transform_multispectral=None, transform_thermal=None,
                 subsample_size=None, split_name='train', split_id=0, 
                verbose=True, resized=True, return_dates=False, standardize_path=None):
        # The fused bags are composed of both multispectral images and thermal images
        self.return_dates = return_dates

        self.multispectral_dataset = CropBagDataset(csv_path, base_path_multispectral, transform=transform_multispectral, 
                                                    subsample_size=subsample_size, split_name=split_name, 
                                                    split_id=split_id, verbose=verbose, resized=resized, 
                                                    eturn_dates=return_dates, standardize_path=standardize_path)
        self.thermal_dataset = ThermalBags(csv_path, base_path_thermal, transform=transform_thermal,
                                                    subsample_size=subsample_size, split_name=split_name, 
                                                    split_id=split_id, return_dates=return_dates)

    
    def __len__(self):
        return len(self.multispectral_dataset)

    def __getitem__(self, index):
        if self.return_dates:
            bag_multispectral, label, dates_multispectral = self.multispectral_dataset[index]
            bag_thermal, _, dates_thermal = self.thermal_dataset[index]
            return bag_multispectral, bag_thermal, label, dates_multispectral, dates_thermal
        else:
            bag_multispectral, label = self.multispectral_dataset[index]
            bag_thermal, _ = self.thermal_dataset[index]
            return bag_multispectral, bag_thermal, label

def variable_length_collate_fusedbags(batch):
    """Combine data with variable length using padding.
    Further add a length tensor to the output, such that the invalid values can
    be appropriately masked.
    Args:
        batch: Nested list of instances
    Returns:
        data, labels, lengths tensors
    """

    imgs_bags = {'multispectral':[], 'thermal': []}
    labels = []
    genotypes = []
    return_geno = False
    dates_bag = {'multispectral':[], 'thermal': []}

    for batch_element in batch:
        # Check if dates are also returned or only imgs and label
        # We cannot do this with an argument
        if len(batch_element) % 2 == 0:
            # Genotypes are also returned
            return_geno = True
        if len(batch_element) <= 4:
            return_dates = False
            if return_geno:
                img_multispectral, img_thermal, geno, label = batch_element
            else:
                img_multispectral, img_thermal, label = batch_element
        elif len(batch_element) <= 6:
            return_dates = True
            if return_geno:
                img_multispectral, img_thermal, geno, label, dates_multispectral, dates_thermal = batch_element
            else:
                img_multispectral, img_thermal, label, dates_multispectral, dates_thermal = batch_element
            dates_bag['multispectral'].append(dates_multispectral)
            dates_bag['thermal'].append(dates_thermal)
        if return_geno:
            genotypes.append(geno)
        imgs_bags['multispectral'].append(img_multispectral)
        imgs_bags['thermal'].append(img_thermal)
        labels.append(label)

    
    lengths = dict()
    extended_imgs_bags = {'multispectral':[], 'thermal': []}
    for img_type in ['multispectral', 'thermal']:
        lengths[img_type] = list(map(len, imgs_bags[img_type]))
        max_len = max(lengths[img_type])

        for imgs, length in zip(imgs_bags[img_type], lengths[img_type]):
            padding = max_len - length
            extended_imgs_bags[img_type].append(
                np.pad(
                    imgs, ((0, padding), (0, 0), (0, 0), (0, 0)),
                    mode='constant'
                )
            )
    if return_dates: 
        extended_dates_bags = {'multispectral':[], 'thermal': []}
        for img_type in ['multispectral', 'thermal']:
            
            max_len = max(lengths[img_type])
            for dates, length in zip(dates_bag[img_type], lengths[img_type]):
                padding = max_len - length
                extended_dates_bags[img_type].append(
                    np.pad(
                        dates, ((0, padding), (0, 0)),
                        mode='constant'
                    )
                )
        if return_geno:
            return tensor(extended_imgs_bags['multispectral'], dtype=torch.float), \
                tensor(extended_imgs_bags['thermal'], dtype=torch.float), \
                tensor(genotypes, dtype=torch.float), \
                tensor(labels, dtype=torch.float),             \
                tensor(lengths['multispectral'], dtype=torch.long),              \
                tensor(lengths['thermal'], dtype=torch.long),               \
                tensor(extended_dates_bags['multispectral'], dtype=torch.float), \
                tensor(extended_dates_bags['thermal'], dtype=torch.float)
        else:
            return tensor(extended_imgs_bags['multispectral'], dtype=torch.float), \
                tensor(extended_imgs_bags['thermal'], dtype=torch.float), \
                tensor(labels, dtype=torch.float),             \
                tensor(lengths['multispectral'], dtype=torch.long),              \
                tensor(lengths['thermal'], dtype=torch.long),               \
                tensor(extended_dates_bags['multispectral'], dtype=torch.float), \
                tensor(extended_dates_bags['thermal'], dtype=torch.float)
    else:
        if return_geno:
            return tensor(extended_imgs_bags['multispectral'], dtype=torch.float), \
                tensor(extended_imgs_bags['thermal'], dtype=torch.float), \
                tensor(genotypes, dtype=torch.float),             \
                tensor(labels, dtype=torch.float),             \
                tensor(lengths['multispectral'], dtype=torch.long),              \
                tensor(lengths['thermal'], dtype=torch.long)
        else:    
            return tensor(extended_imgs_bags['multispectral'], dtype=torch.float), \
                tensor(extended_imgs_bags['thermal'], dtype=torch.float), \
                tensor(labels, dtype=torch.float),             \
                tensor(lengths['multispectral'], dtype=torch.long),              \
                tensor(lengths['thermal'], dtype=torch.long)

#########################################
# Genotype integration
#########################################

class GenotypeDataset(Dataset):
    """Dataset for genotypes in csv files.
    Arguments:
        A CSV file path
        A CSV
    """
    def __init__(self, csv_path, base_path, split_name='train', split_id=0, 
                verbose=True, normalize=True, standardize_target_path=None,
                scaler_path=None):
        # GENOTYPES TO EXCLUDED (because of missingness) were removed in a preprocessing notebook
        # --> use the new CSVs for genotype experiments.
        # The simplest way is to store individual npy arrays for each
        # genotype, and use a mapping from PlotID -> GID in the pytorch dataset. Then, genotypes
        # will be loaded as images, without overloading memory. 
        # PS: The full genotypes are 106 Mb, not excessive, but we stick to single files per samples
        # to mimic image dataset structures.
        self.df = pd.read_csv(csv_path)
        self.split_id = split_id
        self.split_name = split_name
        self.csv_path = csv_path
        self.base_path = base_path
        self.normalize=normalize
        self.standardize_target = False


        if self.normalize:
            if verbose:
                print('Loading training genotypes for normalization...')
            if scaler_path is None:
                # Load all training genotypes and set transformer
                train_idx = self._get_split_from_config('train')
                X_train = np.array([np.load(f'{base_path}/{x}.npy') for x in self.df.iloc[train_idx]['gid']]).astype(np.float32)
                self.scaler = StandardScaler().fit(X_train)
            else:
                self.scaler = pkl.load(open(scaler_path, 'rb'))
            if verbose:
                print('Done.')
        
        # Load standardization values for target
        if standardize_target_path is not None:
            self.standardize_target = True
            df_standardized = pd.read_csv(standardize_target_path, index_col=0)
            self.target_mean = df_standardized.iloc[self.split_id]['mean_yield']
            self.target_std = df_standardized.iloc[self.split_id]['std_yield']
    
        # Load the split files and only keep the relevant split
        idx = self._get_split_from_config()
        self.df = self.df.iloc[idx]
        # Reset the index for later querrying
        self.df.reset_index(inplace=True)

        self.labels_list = list(self.df['GRYLD'].astype(np.float32))

        self.X = base_path + '/' + self.df['gid'].astype(str) + '.npy'
        assert self.X.apply(lambda x: os.path.isfile(x)).all(), "Some genotypes referenced in the CSV file were not found."

    def _get_split_from_config(self, split_name=None):
        json_file = os.path.splitext(self.csv_path)[0] + f'_splits.json'
        config_file = os.path.join(json_file)
        with open(config_file, 'r') as f:
            config = json.load(f)
        if split_name is None:
            return config[str(self.split_id)][self.split_name]
        else:
            return config[str(self.split_id)][split_name]

    def X(self):
        return self.X

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        genotype = np.load(self.X[index]).astype(np.float32)
        label = from_numpy(np.array(self.labels_list[index]).reshape(-1).astype(np.float32))
        if self.normalize:
            genotype = self.scaler.transform(genotype.reshape(1, -1)).reshape(-1)
        # genotype = from_numpy(genotype)
        if self.standardize_target:
            label = (label - self.target_mean)/self.target_std
        return genotype, label

    def getDF(self):
        return self.df
    

class FusedGenoBags(Dataset):
    """Dataset wrapping multispectral images, thermal images, and genotypes.
    Arguments:
        A CSV file path
        A base path where the files are located
        A set of pytorch transformation
        (Optional) a maximum number of samples sampled per bag
    """
    def __init__(self, csv_path, base_path_multispectral, base_path_thermal,
                base_path_genotype, transform_multispectral=None, transform_thermal=None, 
                normalize_genotype=True, subsample_size=None, split_name='train', split_id=0,
                verbose=True, resized=True, return_dates=False, standardize_path=None):
        # The fused bags are composed of both multispectral images and thermal images
        self.return_dates = return_dates

        self.multispectral_dataset = CropBagDataset(csv_path, base_path_multispectral, transform=transform_multispectral, 
                                                    subsample_size=subsample_size, split_name=split_name, 
                                                    split_id=split_id, verbose=verbose, resized=resized, 
                                                    return_dates=return_dates, standardize_path=standardize_path)
        self.thermal_dataset = ThermalBags(csv_path, base_path_thermal, transform=transform_thermal,
                                                    subsample_size=subsample_size, split_name=split_name, 
                                                    split_id=split_id, return_dates=return_dates)
        self.genotype_dataset = GenotypeDataset(csv_path, base_path_genotype, split_name=split_name, split_id=split_id,
                                                    verbose=verbose, normalize=normalize_genotype)

    
    def __len__(self):
        return len(self.multispectral_dataset)

    def __getitem__(self, index):
        # TODO: consider returning a dictionary
        if self.return_dates:
            bag_multispectral, label, dates_multispectral = self.multispectral_dataset[index]
            bag_thermal, _, dates_thermal = self.thermal_dataset[index]
            genotype, _ = self.genotype_dataset[index]
            return bag_multispectral, bag_thermal, genotype, label, dates_multispectral, dates_thermal
        else:
            bag_multispectral, label = self.multispectral_dataset[index]
            bag_thermal, _ = self.thermal_dataset[index]
            genotype, _ = self.genotype_dataset[index]
            return bag_multispectral, bag_thermal, genotype, label

class FusedFullBags(Dataset):
    """Dataset wrapping multispectral images, thermal images, DEM images, and genotypes.
    Arguments:
        A CSV file path
        A base path where the files are located
        A set of pytorch transformation
        (Optional) a maximum number of samples sampled per bag
    """
    def __init__(self, csv_path, base_path_multispectral, base_path_thermal, base_path_dem,
                base_path_genotype, transform_multispectral=None, transform_thermal=None, 
                normalize_genotype=True, subsample_size=None, split_name='train', split_id=0,
                verbose=True, resized=True, return_dates=False, standardize_path=None,
                standardize_target_path=None):
        # The fused bags are composed of both multispectral images and thermal images
        self.return_dates = return_dates

        self.multispectral_dataset = CropBagDataset(csv_path, base_path_multispectral, transform=transform_multispectral, 
                                                    subsample_size=subsample_size, split_name=split_name, 
                                                    split_id=split_id, verbose=verbose, resized=resized, 
                                                    return_dates=return_dates, standardize_path=standardize_path,
                                                    standardize_target_path=standardize_target_path)
        self.thermal_dataset = ThermalBags(csv_path, base_path_thermal, transform=transform_thermal,
                                                    subsample_size=subsample_size, split_name=split_name, 
                                                    split_id=split_id, return_dates=return_dates)
        # Can use the same class for DEM images
        self.dem_dataset = ThermalBags(csv_path, base_path_dem, transform=None,
                                                    subsample_size=subsample_size, split_name=split_name, 
                                                    split_id=split_id, return_dates=return_dates, date_col_name='dates_dem_enc')
        self.genotype_dataset = GenotypeDataset(csv_path, base_path_genotype, split_name=split_name, split_id=split_id,
                                                    verbose=verbose, normalize=normalize_genotype)

    
    def __len__(self):
        return len(self.multispectral_dataset)

    def __getitem__(self, index):
        # TODO: consider returning a dictionary
        if self.return_dates:
            bag_multispectral, label, dates_multispectral = self.multispectral_dataset[index]
            bag_thermal, _, dates_thermal = self.thermal_dataset[index]
            bag_dem, _, dates_dem = self.dem_dataset[index]
            genotype, _ = self.genotype_dataset[index]
            return bag_multispectral, bag_thermal, bag_dem, genotype, label, dates_multispectral, dates_thermal, dates_dem
        else:
            bag_multispectral, label = self.multispectral_dataset[index]
            bag_thermal, _ = self.thermal_dataset[index]
            bag_dem, _ = self.dem_dataset[index]
            genotype, _ = self.genotype_dataset[index]
            return bag_multispectral, bag_thermal, bag_dem, genotype, label

def variable_length_collate_fullfusedbags(batch):
    """Combine data with variable length using padding.
    Further add a length tensor to the output, such that the invalid values can
    be appropriately masked.
    Args:
        batch: Nested list of instances
    Returns:
        data, labels, lengths tensors
    """

    imgs_bags = {'multispectral':[], 'thermal': [], 'dem': []}
    labels = []
    genotypes = []
    dates_bag = {'multispectral':[], 'thermal': [], 'dem': []}

    for batch_element in batch:
        # Check if dates are also returned or only imgs and label
        # We cannot do this with an argument
        if len(batch_element) <= 5:
            return_dates = False
            img_multispectral, img_thermal, img_dem, geno, label = batch_element
        elif len(batch_element) <= 8:
            return_dates = True
            img_multispectral, img_thermal, img_dem, geno, label, dates_multispectral, \
                dates_thermal, dates_dem = batch_element
            dates_bag['multispectral'].append(dates_multispectral)
            dates_bag['thermal'].append(dates_thermal)
            dates_bag['dem'].append(dates_dem)
        genotypes.append(geno)
        imgs_bags['multispectral'].append(img_multispectral)
        imgs_bags['thermal'].append(img_thermal)
        imgs_bags['dem'].append(img_dem)
        labels.append(label)

    
    lengths = dict()
    extended_imgs_bags = {'multispectral':[], 'thermal': [], 'dem': []}
    for img_type in ['multispectral', 'thermal', 'dem']:
        lengths[img_type] = list(map(len, imgs_bags[img_type]))
        max_len = max(lengths[img_type])

        for imgs, length in zip(imgs_bags[img_type], lengths[img_type]):
            padding = max_len - length
            extended_imgs_bags[img_type].append(
                np.pad(
                    imgs, ((0, padding), (0, 0), (0, 0), (0, 0)),
                    mode='constant'
                )
            )
    if return_dates: 
        extended_dates_bags = {'multispectral':[], 'thermal': [], 'dem': []}
        for img_type in ['multispectral', 'thermal', 'dem']:
            
            max_len = max(lengths[img_type])
            for dates, length in zip(dates_bag[img_type], lengths[img_type]):
                padding = max_len - length
                extended_dates_bags[img_type].append(
                    np.pad(
                        dates, ((0, padding), (0, 0)),
                        mode='constant'
                    )
                )
        return tensor(extended_imgs_bags['multispectral'], dtype=torch.float), \
            tensor(extended_imgs_bags['thermal'], dtype=torch.float), \
            tensor(extended_imgs_bags['dem'], dtype=torch.float), \
            tensor(genotypes, dtype=torch.float), \
            tensor(labels, dtype=torch.float),             \
            tensor(lengths['multispectral'], dtype=torch.long),              \
            tensor(lengths['thermal'], dtype=torch.long),                \
            tensor(lengths['dem'], dtype=torch.long), \
            tensor(extended_dates_bags['multispectral'], dtype=torch.float), \
            tensor(extended_dates_bags['thermal'], dtype=torch.float), \
            tensor(extended_dates_bags['dem'], dtype=torch.float)
    else:
        return tensor(extended_imgs_bags['multispectral'], dtype=torch.float), \
            tensor(extended_imgs_bags['thermal'], dtype=torch.float), \
            tensor(extended_imgs_bags['dem'], dtype=torch.float), \
            tensor(genotypes, dtype=torch.float),             \
            tensor(labels, dtype=torch.float),             \
            tensor(lengths['multispectral'], dtype=torch.long),              \
            tensor(lengths['thermal'], dtype=torch.long),                \
            tensor(lengths['dem'], dtype=torch.long)
