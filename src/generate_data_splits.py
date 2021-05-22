'''
Script that generates the reproducible splits used in the graphkernels 
evaluation
'''
import numpy as np
import os
import pandas as pd
import json
import argparse
import itertools

from sklearn.model_selection import GroupKFold


def main():
    """Iterate over all datasets and generate splits"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'ROOT',
        type=str,
        help='Root directory containing all datasets'
    )
    args = parser.parse_args()

    # Load base file
    csv_name = 'df_20191014_numpy_MIL_npy_coordinates'
    # csv_name_short = csv_name + '_MINI'
    
    # for name in [csv_name, csv_name_short]:
    for name in [csv_name]:
        print(f'Generating splits for {name}')
        df_filepaths = pd.read_csv(os.path.join(args.ROOT, 'csv', name + '.csv'))


        # Check the trial split
        n_trial = df_filepaths['tid'].nunique()
        n_valtest = round(0.1*n_trial)
        print(f'There are {n_trial} individual trials, there will be {n_valtest} trials for testing')

        # Set the seed
        np.random.seed(42)

        # Pgroupout crossvalidation
        gkf = GroupKFold(n_splits=10)

        split_dict = dict()

        # run through the folds
        for fold, (train_index, test_index) in enumerate(gkf.split(df_filepaths, groups=df_filepaths['tid'])):            
            print("Computing results for groupkfold fold {}.".format(fold))
            split_dict[fold] = dict()

            # Generate new files
            # df_train, df_test = train_test_split(df_filepaths, test_size=crossvalidation['test_size'], stratify=df_filepaths['Path'])
            # df_train, df_val = train_test_split(df_train, test_size=crossvalidation['val_size'], stratify=df_train['Path'])
            df_train = df_filepaths.iloc[train_index]

            gkf_internal = GroupKFold(n_splits=9)
            train_train_index, val_index = next(gkf_internal.split(train_index, groups=df_train['tid']))
            train_train_index = train_index[train_train_index]
            val_index = train_index[val_index]

            split_dict[fold]['val'] = val_index.astype(int).tolist()
            split_dict[fold]['train'] = train_train_index.astype(int).tolist()
            split_dict[fold]['test'] = test_index.astype(int).tolist()
            print(len(split_dict[fold]['val']))
            print(len(split_dict[fold]['train']))
            for split_a, split_b in itertools.permutations(['train', 'val', 'test'], 2):
                assert len(set(split_dict[fold][split_a]).intersection(set(split_dict[fold][split_b]))) == 0

        # Save to file
        filename = os.path.join(args.ROOT, 'csv', f'{name}_splits.json')
        print(filename)
        with open(filename, 'w') as fp:
            json.dump(split_dict, fp)
    
if __name__ == '__main__':
    main()