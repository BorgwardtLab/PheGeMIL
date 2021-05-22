import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


DATA_FOLDER='/links/groups/borgwardt/Data/Jesse_2018/'
df_vi = pd.read_csv(os.path.join(DATA_FOLDER, 'csv', 'df_20190718_numpy_coordinates_VI.csv'))
df_vi=df_vi.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'iyear', 'ilocation', 'itrial','icondition',
                           'site','year', 'location', 'cycle', 'conditions'])
FEATURE_COLUMNS = ['ndvi_min','ndvi_max', 'ndvi_mean', 'ndvi_std', 'gndvi_min', 'gndvi_max',
       'gndvi_mean', 'gndvi_std', 'rendvi_min', 'rendvi_max', 'rendvi_mean',
       'rendvi_std', 'endvi_min', 'endvi_max', 'endvi_mean', 'endvi_std',
       'gipvi_min', 'gipvi_max', 'gipvi_mean', 'gipvi_std']


# Use splits used by DL models
def get_train_test_split(sacred_id):
    # Retrieve the number of folds
    sacred_path = os.path.join(DATA_FOLDER, 'output/logs/sacred_logs/crops_new_data_vanilla_yield', 
                               str(sacred_id))
    with open(os.path.join(sacred_path, 'config.json'), 'r') as c:
        config = json.load(c)
    indeces = dict()
    for fold in range(config['crossvalidation']['folds']):
        indeces[fold] = dict()
        # load dfs
        for split in ['train', 'val', 'test']:
            indeces[fold][split] = set(pd.read_csv(os.path.join(sacred_path, '{}_fold{}.csv'.format(split,fold)))['Filename'])
    return indeces

# Split the data and create X and y
def split_dataset(df_generic, index_set):
    # Get training and validation data
    df = df_generic.set_index('Filename')
    dfs = dict()
    dfs_filtered = dict()
    rmvd = dict()
    tot_rem = 0
    for split in ['train', 'val', 'test']:
        dfs[split] = df.loc[list(index_set[split])]
        # Since most methods require no NaNs, values need to be taken out
        dfs_filtered[split] = dfs[split][~dfs[split][FEATURE_COLUMNS].isna().any(axis=1)]
        tot_rem += dfs[split][FEATURE_COLUMNS].isna().any(axis=1).sum()
        rmvd[split] = 100*dfs[split][FEATURE_COLUMNS].isna().any(axis=1).sum()/len(dfs[split])
    print("There are NaN values, {:.2f} % of training samples, {:.2f} % of validation, and {:.2f} % of test samples had to be removed ({} samples in total).".format(
            rmvd['train'], rmvd['val'], rmvd['test'],tot_rem))
    return dfs_filtered['train'], dfs_filtered['val'], dfs_filtered['test']

# function to get X and y
def get_xy_from_vis(df, w_coordinates=False):
    # Returns x and y from df
    if w_coordinates:
        feats = FEATURE_COLUMNS + ['coordinates_x', 'coordinates_y']
    else:
        feats = FEATURE_COLUMNS
    x = df[feats].values
    y = df['GRYLD'].values
    return x,y

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
                                   n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs=4)
    print('Fitting the model, this may take some time.')
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    return rf_random

def main():
    np.random.seed(42)
    indeces = get_train_test_split(47)
    for i in range(len(indeces)):
        # Retrieve the splitted data
        print('Split {}'.format(i))
        df_train, df_val, df_test = split_dataset(df_vi, indeces[i])
        
        # Obtain X and y
        X_train, y_train = get_xy_from_vis(pd.concat([df_train,df_val]))
        X_test, y_test = get_xy_from_vis(df_test)
        
        print(np.isnan(X_train).any())
        
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Hyperparameters search
        rf = train_random_forest(X_train, y_train)
        print('Model fitted.')
        # Evaluation
        y_pred = rf.predict(X_test)
        df_test['pred'] = y_pred
        df_test.to_csv('../../results/baselines_201907/rf_split{}.csv'.format(i), index=False)
        r2_sc = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print('MSE: {:.4f}, R2_SCORE: {:.4f}'.format(mse, r2_sc))
        print()
    
    return

if __name__== '__main__':
    main()
    