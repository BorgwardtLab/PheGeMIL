import numpy as np
import pandas as pd
import os
import json
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score

# def parallel_comp(model, K, y, train_index, test_index, param_grid):
#     # Run over parameters
#     res = []
#     for p in list(ParameterGrid(param_grid)):
#         sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(r2_score), 
#                 train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
#         res.append(sc)
#         params.append({'K_idx': K_idx, 'params': p})
#     return res

def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, trial_ids, cv=5):
    '''
    Custom grid search based on the sklearn grid search for an array of precomputed kernels and trial ID strat
    '''
    # 1. Stratified K-fold
    cv = GroupKFold(n_splits=cv)
    results = []
    print("Started custom CV")
    for train_index, test_index in tqdm(cv.split(precomputed_kernels[0], y, groups = trial_ids)):
        split_results = []
        params = [] # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(r2_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc)
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    print("Custom CV done.")
    print()
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]


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

def get_split_idx(df_m, split_id=0):
    # Get training and validation data
    split_dict = split_plot_ids(split_id=split_id)
    
    idxs = dict()
    
    # First filter and only keep entries from the relevant date
    print(f"There are {len(df_m)} entries.")
    for split in ['train', 'val', 'test']:
        split_idx = [i for (i,plotid) in enumerate(df_m['PlotID']) if plotid in split_dict[split]]
    
        idxs[split] = split_idx
    return idxs


def train_random_forest(X_train, y_train):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                   n_iter = 50, cv = 3, verbose=1, random_state=42, n_jobs=4)
    print('Fitting the model, this may take some time.')
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    return rf_random

def main():
    df_mil = pd.read_csv('/links/groups/borgwardt/Data/Jesse_2018/csv/df_20200121_numpy_MIL_npy_coordinates_dates_genofiltered_dems.csv')
    results = []
    split_id_range = 5
    # for date_group in [1,2,3,4,'all']: # TBD
    for date_group in ['all',1]: # TBD
        # Load Wasserstein matrix
        # X = np.load(f'/links/groups/borgwardt/Data/Jesse_2018/wass_matrices/full_wass_dist_dategroup_{date_group}.npy')
        X = np.load(f'/links/groups/borgwardt/Data/Jesse_2018/wass_matrices_hd/full_wass_dist_dategroup_{date_group}.npy')
        # Remove samples with missing entries if necessary
        idxs_to_drop = np.where(np.isnan(X).all(axis=1))[0]
        if len(idxs_to_drop)>0:
            df_mil_filtered = df_mil.drop([df_mil.index[x] for x in idxs_to_drop])
            X = np.delete(X, idxs_to_drop, 0)
            X = np.delete(X, idxs_to_drop, 1)
        else:
            df_mil_filtered = df_mil
        print(f'=============== Wasserstein Kernels - date group {date_group} ===============')
        for i in range(split_id_range):
            # Retrieve the splitted data
            print(f'Wkernels - Split {i}')
            split_idx = get_split_idx(df_mil_filtered, split_id=i)

            # Combine train and val
            split_idx['train'] = sorted(split_idx['train']+split_idx['val'])

            # Parameter lists
            gammas = [0.001, 0.01, 0.1, 1]
            param_grid = [
                    {'C': np.logspace(-3,3,num=7)}
                ]
            
            # Obtain Ks, y and tid
            Ks = [np.exp(-gamma*X) for gamma in gammas]
            y = df_mil_filtered['GRYLD'].values
            tid = df_mil_filtered['tid'].values

            # Train test split
            K_trains = [K[split_idx['train']][:, split_idx['train']] for K in Ks]
            y_train = y[split_idx['train']]
            tid_train = tid[split_idx['train']]
            print(f'{len(y_train)} training examples.')

            # Hyperopt and fit
            trained_svr, best_params = custom_grid_search_cv(SVR(kernel='precomputed'), param_grid, K_trains, y_train, tid_train)
            print('Best params:')
            print(best_params)

            # Test and eval
            K_test = Ks[best_params['K_idx']][split_idx['test']][:, split_idx['train']] 
            y_pred = trained_svr.predict(K_test)
            y_test = y[split_idx['test']]

            r2_sc = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            pears = pearsonr(y_test, y_pred)[0]

            results.append(['kwasserstein', date_group, i, mae, mse, r2_sc, pears])
            print('MSE: {:.4f}, R2_SCORE: {:.4f}'.format(mse, r2_sc))
            print()
        print()
        print()
        # Save results
        # pd.DataFrame(results, columns=['method', 'date_group', 'split_id', 'mae', 'mse', 'r2_sc', 'pears']).to_csv(
        #     f'../../results/baselines_20191115/wasserstein_results_dategroup{date_group}.csv', index=False
        # )
        pd.DataFrame(results, columns=['method', 'date_group', 'split_id', 'mae', 'mse', 'r2_sc', 'pears']).to_csv(
            f'../../results/baselines_20191115/wasserstein_hd_results_dategroup{date_group}.csv', index=False
        )
    return

if __name__== '__main__':
    main()
    
