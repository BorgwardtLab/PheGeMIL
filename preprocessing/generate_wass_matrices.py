import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import ot

def main():
    data_path = '/links/groups/borgwardt/Data/Jesse_2018'
    df_mil = pd.read_csv(os.path.join(data_path,
                                    'csv',
                                    'df_20200121_numpy_MIL_npy_coordinates_dates_genofiltered_dems.csv'))
    
    # for date_group in [1,2,3,4,'all']: 
    for date_group in [1,4]: 
        # Iterate through the samples in the order AS THEY APPEAR IN THE CSV FILE
        hist_folder = os.path.join(data_path, 'numpy_MIL_resized', 'histograms_ms', f'date_group_{date_group}')
        # 1. load histograms
        hists = []
        for i, row in tqdm(df_mil.iterrows()):
            hists.append(np.load(os.path.join(hist_folder, row['Filename'])).reshape(-1,1))

        # 2. compute the pairwise distances:
        n = len(hists)
        M = np.zeros((n,n))

        for i, hist_1 in tqdm(enumerate(hists)):
            for j, hist_2 in enumerate(hists[i:]):
                M[i, i+j] = ot.emd2_1d(hist_1, hist_2, metric='euclidean')
        
        M = M + M.T
        np.save(os.path.join(data_path, 'wass_matrices', f'full_wass_dist_dategroup_{date_group}.npy'), M)
        print(f'Wasserstein matrix for Date Group {date_group} completed.')


if __name__ == "__main__":
    main()