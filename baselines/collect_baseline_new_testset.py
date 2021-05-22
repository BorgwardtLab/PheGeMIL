# ------------------------------------------
# Quick script to compute performance of 
# linear baselines
# ------------------------------------------
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pickle

def main():
    output_file = ('/home/tomatteo/Projects/yield_prediction/results/baselines_202012/lassocv_performance.csv')

    # Load the genotypes of new plots.
    df_inference = pd.read_csv('/links/groups/borgwardt/Data/Jesse_2018/202010_new_testset/csv/'\
                            'df_20201110_numpy_MIL_npy_coordinates_geno_filtered.csv')
    # Load genotypes
    base_path = '/links/groups/borgwardt/Data/Jesse_2018/numpy_MIL_resized/genotypes'

    # Remove known genotypes
    old_gids = np.loadtxt('/links/groups/borgwardt/Data/Jesse_2018/genotypes/gids.txt', dtype=str)

    index = df_inference.query(
        "gid not in @old_gids")['gid'].unique()

    X_inference = np.array([np.load(f'{base_path}/{x}.npy') for x in index]).astype(np.float32)

    # Load trained model and scalers
    output_path = '/home/tomatteo/Projects/yield_prediction/results/baselines_202012/'

    genotype_yield_dict_inference = df_inference.groupby('gid')['GRYLD'].mean().to_dict()
    y_test = [genotype_yield_dict_inference[idx] for idx in index]

    results = []
    # Run across folds
    for i in range(5):
        yscaler = pickle.load(open(os.path.join(output_path,f'y_scaler_fold{i}.pkl'), 'rb'))
        scaler = pickle.load(open(os.path.join(output_path,f'scaler_fold{i}.pkl'), 'rb'))
        model = pickle.load(open(os.path.join(output_path,f'trained_lassocv_fold{i}.pkl'), 'rb'))

        X_inference_scaled = scaler.transform(X_inference)
        y_pred = yscaler.inverse_transform(model.predict(X_inference_scaled))
        y_test = [genotype_yield_dict_inference[idx] for idx in index]

        # Compute performance
        r2_sc = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pears = pearsonr(y_test, y_pred)[0]

        print('MSE: {:.4f}, R2_SCORE: {:.4f}, Pears: {:.4f}'.format(mse, r2_sc, pears))
        print()
        results.append(['genotype_lassocv', i, mae, mse, r2_sc, pears])

    pd.DataFrame(results, columns=['method', 'split_id', 'mae', 'mse', 'r2_sc', 'pears']).to_csv(
            output_file, index=False
        )

if __name__== '__main__':
    main()
