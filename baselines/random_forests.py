import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


FEATURE_COLUMNS = ['BLUE_Mean', 'BLUE_Mode', 'GNDVI_Mean', 'GNDVI_Mode',
       'GREEN_Mean', 'GREEN_Mode', 'NDRE_Mean', 'NDRE_Mode', 'NDVI_Mean',
       'NDVI_Mode', 'NIR_Mean', 'NIR_Mode', 'REDEDGE_Mean', 'REDEDGE_Mode',
       'RED_Mean', 'RED_Mode']

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

# Split the data and create X and y
def split_dataset(master_table, split_id=0, date_group=1):
    # Get training and validation data
    split_dict = split_plot_ids(split_id=split_id)
    
    dfs = dict()
    dfs_filtered = dict()
    # No need to keep validation data
    if date_group == 'all':
        df_m = master_table
    else:
        df_m = master_table[master_table['date_group']==date_group]
    # First filter and only keep entries from the relevant date
    print(f"There are {len(df_m)} entries for date_group {date_group}")
    for split in ['train', 'val', 'test']:
        split_mask = [plotid in split_dict[split] for plotid in df_m['Plot_ID']]
    
        dfs[split] = df_m[split_mask]
#         print(f'There are {len(dfs[split])} entries for the {split} split')
    return dfs['train'], dfs['val'], dfs['test']

def get_xy_from_vis(df): 
    # Returns x and y from df
    x = df[FEATURE_COLUMNS].values
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
                                   n_iter = 50, cv = 3, verbose=1, random_state=42, n_jobs=4)
    print('Fitting the model, this may take some time.')
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    return rf_random

def main():
    master_table = pd.read_csv('/links/groups/borgwardt/Data/Jesse_2018/csv/df_20191114_VIs.csv')
    results = []
    split_id_range = 5
    for date_group in [1,2,3,4,'all']:
        print(f'=============== randomforest - date group {date_group} ===============')
        for i in range(split_id_range):
            # Retrieve the splitted data
            print(f'randomforest - Split {i}')
            df_train, df_val, df_test = split_dataset(master_table, split_id=i, date_group=date_group)

            # Obtain X and y
            X_train, y_train = get_xy_from_vis(pd.concat([df_train,df_val]))
            print(f'{len(X_train)} training examples.')
            X_test, y_test = get_xy_from_vis(df_test)

            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # HYPEROPT
            rf = train_random_forest(X_train, y_train)
            print('Model fitted.')
            # Evaluation
            y_pred = rf.predict(X_test)

            r2_sc = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            pears = pearsonr(y_test, y_pred)[0]

            results.append(['randomforest', date_group, i, mae, mse, r2_sc, pears])
            print('MSE: {:.4f}, R2_SCORE: {:.4f}'.format(mse, r2_sc))
            print()
        print()
        print()
    # Save results
    pd.DataFrame(results, columns=['method', 'date_group', 'split_id', 'mae', 'mse', 'r2_sc', 'pears']).to_csv(
        f'../../results/baselines_20191115/randomforest_results.csv', index=False
    )
    return

if __name__== '__main__':
    main()
    