"""
Contains the script for evaluating the baseline models
On the newly unseen data
Preprocessing done in Notebook "2020.12.3_genotype_baselines_newdataset.ipynb"

M. Togninalli, 01.2021
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LassoCV

from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
import pickle

# Reduced set of features available for newer data set, to get same order
FEATURE_COLUMNS = ['BLUE_Mean', 'GNDVI_Mean',
       'GREEN_Mean', 'NDRE_Mean', 'NDVI_Mean',
       'NIR_Mean','REDEDGE_Mean', 'RED_Mean']

# Train and test
def get_xy_df(df):
    x = df.drop(columns=['GRYLD', 'Plot_ID']).values
    # x = df[FEATURE_COLUMNS].values
    y = df['GRYLD'].values
    return x,y

def main():
    # OPTION:
    optimize_params = True

    # Load the genotypes of new plots.
    df_inference = pd.read_csv('/links/groups/borgwardt/Data/Jesse_2018/202010_new_testset/csv/'\
                            'df_20201110_numpy_MIL_npy_coordinates_geno_filtered.csv')
    # 2. Load genotypes
    # Alternative way of loading genotypes
    base_path = '/links/groups/borgwardt/Data/Jesse_2018/numpy_MIL_resized/genotypes'
    # Remove known genotypes
    old_gids = np.loadtxt('/links/groups/borgwardt/Data/Jesse_2018/genotypes/gids.txt', dtype=str)
    
    index = df_inference.query(
        "gid not in @old_gids")['gid'].unique()
    X = np.array([np.load(f'{base_path}/{x}.npy') for x in df_inference.query(
        "gid not in @old_gids")['gid'].unique()]).astype(np.float32)
    df_genotype_processed = pd.DataFrame(X, index=index)


    # 3. Load VIs
    df_vis = pd.read_csv('/links/groups/borgwardt/Data/Jesse_2018/202010_new_testset/csv/df_20210125_VIs.csv')
    date_group = 2 # Change this potentially
    # Only keep the relevant group
    df_vis = df_vis[df_vis['date_group']==date_group]

    # Concatenate the dataframes
    df_vis['gid'] = df_vis['Plot_ID'].map(df_inference.set_index('plot_id').gid)
    df_vis = df_vis.dropna()
    df_vis = df_vis.join(df_genotype_processed, on='gid', how='left').dropna()
    # Remove unnecessary infor
    gids = df_vis['gid']
    plot_ids = df_vis['Plot_ID']
    df_vis = df_vis.drop(columns=['date_group', 'gid'])
    del df_genotype_processed
    print(f'DFs loaded and concatenated: dims {df_vis.shape}')
    
    results = []
    split_id_range = 5
    
    # Obtain X and y
    X_test, y_test = get_xy_df(df_vis)
    print(f'{len(X_test)} inference samples.')

    # Iterate over splits
    output_path = f'/home/tomatteo/Projects/yield_prediction/results/baselines_202012/VIs'
    for i in range(split_id_range):
        # Retrieve the splitted data
        print(f'=============== VIs - Genotype baseline - Data Preprocessing - split {i} ===============')
        yscaler = pickle.load(open(os.path.join(output_path,f'y_scaler_fold{i}.pkl'), 'rb'))
        scaler = pickle.load(open(os.path.join(output_path,f'scaler_fold{i}.pkl'), 'rb'))
        lasso = pickle.load(open(os.path.join(output_path,f'trained_geno_vis_lasso_fold{i}_hyperopt_{optimize_params}.pkl'), 'rb'))
        rf = pickle.load(open(os.path.join(output_path,f'trained_geno_vis_rf_fold{i}_hyperopt_{optimize_params}.pkl'), 'rb'))

        X_test = scaler.transform(X_test)

        print('\tData loaded and scaled.')

        print(f'=============== VIs - Genotype baseline - Random Forest - split {i} ===============')
        # Evaluation
        y_pred = yscaler.inverse_transform(rf.predict(X_test))

        r2_sc = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pears = pearsonr(y_test, y_pred)[0]

        results.append(['full_randomforest', i, mae, mse, r2_sc, pears, date_group])
        print('MSE: {:.4f}, R2_SCORE: {:.4f}, Pearson: {:.4f}'.format(mse, r2_sc, pears))
        print()

        pd.DataFrame(np.array([plot_ids,gids,y_test,y_pred]).T, columns=['plot_id','gid','target_yield', 'predicted_yield']).to_csv(
                 os.path.join(output_path, f'results_rf_hyperopt_{optimize_params}_split{i}.csv'), index=False)

        print(f'=============== VIs - Genotype baseline - Lasso - split {i} ===============')

        y_pred = yscaler.inverse_transform(lasso.predict(X_test))

        r2_sc = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pears = pearsonr(y_test, y_pred)[0]

        results.append(['Lasso', i, mae, mse, r2_sc, pears, date_group])
        print('MSE: {:.4f}, R2_SCORE: {:.4f}, Pearson: {:.4f}'.format(mse, r2_sc, pears))
        print()

        pd.DataFrame(np.array([plot_ids,gids,y_test,y_pred]).T, columns=['plot_id','gis','target_yield', 'predicted_yield']).to_csv(
                 os.path.join(output_path, f'results_lasso_hyperopt_{optimize_params}_split{i}.csv'), index=False)

    # Save results
    pd.DataFrame(results, columns=['method', 'split_id', 'mae', 'mse', 'r2_sc', 'pears', 'date_group']).to_csv(
        f'/home/tomatteo/Projects/yield_prediction/results/baselines_202012/VIs/new_testset_results_20210126_hyperopt_{optimize_params}.csv', index=False
    )
    return

if __name__== '__main__':
    main()
    