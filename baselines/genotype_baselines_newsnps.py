import numpy as np
import pandas as pd
import os
import json

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

import pickle

GIDS_TO_EXCLUDE = {4755014, 7806808} # 8243873 8242572 are not present
MISSING_GENOTYPES_GIDS = {} # All genotypes are present in the new data set 
GENOTYPES_TO_EXCLUE = GIDS_TO_EXCLUDE
# Helper functions
def split_genotype_df(df_genotype, split_dict, gids_to_exclude=None):
    # Get training and testing data
    # Split dict contains the split ids.
    dfs = {}
    for key in list(split_dict.keys()):
        l = list(split_dict[key])
        if gids_to_exclude is not None:
            l = list(set(l)-gids_to_exclude)
        dfs[key] = df_genotype.loc[l]
    return dfs

# Unfortunately, the config file only returns the indeces of the dataset, we need to retrieve the plot_ids from there.
def split_plot_ids(split_id=0):
    # Retrieve the train, val and test plot_ids for a given split_id
    # First, load the original dfs
    base_path = '/links/groups/borgwardt/Data/Jesse_2018/csv/'
    df = pd.read_csv(os.path.join(base_path, 'df_20200121_numpy_MIL_npy_coordinates_dates_genofiltered_dems.csv'))
    json_file = os.path.join(base_path, 'df_20200121_numpy_MIL_npy_coordinates_dates_genofiltered_dems_splits.json')
    # Get config
    with open(json_file, 'r') as f:
        config = json.load(f)[str(split_id)]
    plot_ids = {}
    for split in ['train', 'val', 'test']:
        plot_ids[split] = np.unique(df['PlotID'].iloc[config[split]])
    return plot_ids

# Train and test
def get_xy_from_genodf(df):
    x = df.drop(columns='GRYLD').values
    y = df['GRYLD'].values
    return x,y

def main():
    # 1. Load usual workbook
    df = pd.read_csv('/links/groups/borgwardt/Data/Jesse_2018/csv/df_20200121_numpy_MIL_npy_coordinates_dates_genofiltered_dems.csv')
    # 2. Load genotypes
    #df_genotype_processed = pd.read_csv('/links//groups/borgwardt/Data/Jesse_2018/genotypes/20201028_df_genotypes_2013_2020_full.csv', 
    #                        index_col=0)
    # First of all, remove the genotypes
    # Only keep relevant genotypes
    # df_genotype_processed = df_genotype_processed.loc[df['gid'].unique()]

    # Alternative loading of genotypes
    base_path = '/links/groups/borgwardt/Data/Jesse_2018/numpy_MIL_resized/genotypes'
    index = df['gid'].unique()
    X = np.array([np.load(f'{base_path}/{x}.npy') for x in df['gid'].unique()]).astype(np.float32)
    df_genotype_processed = pd.DataFrame(X, index=index)
    df_genotype_processed = df_genotype_processed.drop(index=GENOTYPES_TO_EXCLUE)

    genotype_yield_dict = df.groupby('gid')['GRYLD'].mean().to_dict()
    df_genotype_processed['GRYLD'] = df_genotype_processed.index.map(genotype_yield_dict)

    results = []
    split_id_range = 5
    
    # Iterate over splits
    for i in range(split_id_range):
        # Retrieve the splitted data
        print(f'Genotype baseline - Split {i}')

        ids = split_plot_ids(i)
        gids_split = dict()
        for split in ['train', 'val', 'test']:
            split_mask = [plotid in ids[split] for plotid in df['PlotID']]
            gids_split[split] = df[split_mask]['gid'].unique()
        
        dfs_dict = split_genotype_df(df_genotype_processed, gids_split, gids_to_exclude=GIDS_TO_EXCLUDE)

        # Obtain X and y
        X_train, y_train = get_xy_from_genodf(pd.concat([dfs_dict['train'],dfs_dict['val']]))
        X_test, y_test = get_xy_from_genodf(dfs_dict['test'])
        print(f'{len(X_train)} training examples.')

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        yscaler = StandardScaler().fit(y_train.reshape(-1,1))
        y_train = yscaler.transform(y_train.reshape(-1,1))

        lassocv = LassoCV(verbose=2., n_alphas=40)
        lassocv.fit(X_train, y_train)
        print('Model fitted.')

        y_pred = yscaler.inverse_transform(lassocv.predict(X_test))

        r2_sc = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pears = pearsonr(y_test, y_pred)[0]

        results.append(['genotype_lassocv', i, mae, mse, r2_sc, pears])
        print('MSE: {:.4f}, R2_SCORE: {:.4f}'.format(mse, r2_sc))
        print()

        # Save scalers and trained model
        output_path = '../../results/baselines_202012/'
        pickle.dump(yscaler, open(os.path.join(output_path,f'y_scaler_fold{i}.pkl'), 'wb'))
        pickle.dump(scaler, open(os.path.join(output_path,f'scaler_fold{i}.pkl'), 'wb'))
        pickle.dump(lassocv, open(os.path.join(output_path,f'trained_lassocv_fold{i}.pkl'), 'wb'))
    # Save results
    pd.DataFrame(results, columns=['method', 'split_id', 'mae', 'mse', 'r2_sc', 'pears']).to_csv(
        f'../../results/baselines_202012/genotype_results_20201203.csv', index=False
    )
    return


if __name__== '__main__':
    main()
    
