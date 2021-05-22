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

GIDS_TO_EXCLUDE = {4755014, 7806808, 8242572, 8243873}
MISSING_GENOTYPES_GIDS = {8242572, 8243873} 
GENOTYPES_TO_EXCLUE = GIDS_TO_EXCLUDE - MISSING_GENOTYPES_GIDS

# Split the data and create X and y
def split_df(df_m, split_id=0):
    # Get training and validation data
    split_dict = split_plot_ids(split_id=split_id)
    
    dfs = dict()
    
    # First filter and only keep entries from the relevant date
    print(f"There are {len(df_m)} entries.")
    for split in ['train', 'val', 'test']:
        split_mask = [plotid in split_dict[split] for plotid in df_m['Plot_ID']]
    
        dfs[split] = df_m[split_mask]
    return dfs

# Unfortunately, the config file only returns the indeces of the dataset, we need to retrieve the plot_ids from there.
def split_plot_ids(split_id=0):
    # Retrieve the train, val and test plot_ids for a given split_id
    # First, load the original dfs
    base_path = '/links/groups/borgwardt/Data/Jesse_2018/csv/'
    df = pd.read_csv(os.path.join(base_path, 'df_20191014_numpy_MIL_npy_coordinates.csv'))
    json_file = os.path.join(base_path, 'df_20191014_numpy_MIL_npy_coordinates_splits.json')
    # Get config
    with open(json_file, 'r') as f:
        config = json.load(f)[str(split_id)]
    plot_ids = {}
    for split in ['train', 'val', 'test']:
        plot_ids[split] = np.unique(df['PlotID'].iloc[config[split]])
    return plot_ids

# Train and test
def get_xy_df(df):
    x = df.drop(columns=['GRYLD', 'Plot_ID']).values
    y = df['GRYLD'].values
    return x,y

def train_random_forest(X_train, y_train):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(40, 80, num = 4)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor(n_jobs=8)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                   n_iter = 5, cv = 3, verbose=1, random_state=42, n_jobs=3)
    print('Fitting the model, this may take some time...')
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    return rf_random

def main():
    # 1. Load usual workbook
    df = pd.read_csv('/links/groups/borgwardt/Data/Jesse_2018/csv/df_20191014_numpy_MIL_npy_coordinates.csv')
    # 2. Load genotypes
    df_genotypes = pd.read_csv('/links//groups/borgwardt/Data/Jesse_2018/genotypes/20191203_df_genotypes.csv', 
                            index_col=0)
    # 3. Load VIs
    df_vis = pd.read_csv('/links/groups/borgwardt/Data/Jesse_2018/csv/df_20191114_VIs.csv')
    date_group = 2
    # Only keep the relevant group
    df_vis = df_vis[df_vis['date_group']==date_group]

    # First of all, remove the genotypes
    df_genotype_processed = df_genotypes.drop(index=GENOTYPES_TO_EXCLUE)

    # Concatenate the dataframes
    df_vis['gid'] = df_vis['Plot_ID'].map(df.set_index('PlotID').gid)
    df_vis = df_vis.dropna()
    df_vis = df_vis.join(df_genotypes, on='gid', how='left').dropna()
    # Remove unnecessary infor
    df_vis = df_vis.drop(columns=['date', 'date_group', 'gid'])
    del df_genotypes
    print(f'DFs loaded and concatenated: dims {df_vis.shape}')


    # genotype_yield_dict = df.groupby('gid')['GRYLD'].mean().to_dict()
    # df_genotype_processed['GRYLD'] = df_genotype_processed.index.map(genotype_yield_dict)

    results = []
    split_id_range = 5
    
    # Iterate over splits
    for i in range(split_id_range):
        # Retrieve the splitted data
        print(f'=============== VIs - Genotype baseline - Data Preprocessing - split {i} ===============')
        # Get Data
        df_dict = split_df(df_vis, split_id=i)

        # Obtain X and y
        X_train, y_train = get_xy_df(pd.concat([df_dict['train'],df_dict['val']]))
        X_test, y_test = get_xy_df(df_dict['test'])
        print(f'{len(X_train)} training examples.')

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        yscaler = StandardScaler().fit(y_train.reshape(-1,1))
        y_train = yscaler.transform(y_train.reshape(-1,1))

        print('\tData Scaled and loaded.')

        print(f'=============== VIs - Genotype baseline - Random Forest - split {i} ===============')
        # HYPEROPT
        rf = train_random_forest(X_train, y_train)
        print('Model fitted.')
        # Evaluation
        y_pred = yscaler.inverse_transform(rf.predict(X_test))

        r2_sc = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pears = pearsonr(y_test, y_pred)[0]

        results.append(['full_randomforest', i, mae, mse, r2_sc, pears, date_group])
        print('MSE: {:.4f}, R2_SCORE: {:.4f}'.format(mse, r2_sc))
        print()

        print(f'=============== VIs - Genotype baseline - Lasso - split {i} ===============')

        lassocv = LassoCV(verbose=2., n_alphas=40, n_jobs=32)
        lassocv.fit(X_train, y_train)
        print('Model fitted.')

        y_pred = yscaler.inverse_transform(lassocv.predict(X_test))

        r2_sc = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pears = pearsonr(y_test, y_pred)[0]

        results.append(['full_lassocv', i, mae, mse, r2_sc, pears, date_group])
        print('MSE: {:.4f}, R2_SCORE: {:.4f}'.format(mse, r2_sc))
        print()

    # Save results
    pd.DataFrame(results, columns=['method', 'split_id', 'mae', 'mse', 'r2_sc', 'pears', 'date_group']).to_csv(
        f'../../results/baselines_20191115/fullbaseline_results_20200121.csv', index=False
    )
    return

if __name__== '__main__':
    main()
    