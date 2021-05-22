import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import ot
from joblib import Parallel, delayed

def parallel_M(hist, otherhists, n):
    M = np.zeros((n))
    offset = n-len(otherhists)
    for i,h in enumerate(otherhists):
        M[offset+i] = ot.emd2_1d(hist, h, metric='euclidean')
    return M


def main():
    data_path = '/links/groups/borgwardt/Data/Jesse_2018'
    df_mil = pd.read_csv(os.path.join(data_path,
                                    'csv',
                                    'df_20200121_numpy_MIL_npy_coordinates_dates_genofiltered_dems.csv'))
    
    # for date_group in [1,2,3,4,'all']: 
    # for date_group in [1,3,4]:
    # for date_group in [2, 'all']:
    for date_group in [3,4]:
        # Iterate through the samples in the order AS THEY APPEAR IN THE CSV FILE
        hist_folder = os.path.join(data_path, 'numpy_MIL_resized', 'histograms_ms_hd', f'date_group_{date_group}')
        # hist_folder = os.path.join(data_path, 'numpy_MIL_resized', 'histograms_ms', f'date_group_{date_group}')
        # 1. load histograms
        hists = []
        for i, row in tqdm(df_mil.iterrows()):
            hists.append(np.load(os.path.join(hist_folder, row['Filename'])).reshape(-1))

        # # Remove all zero entries THERE ARE NONE!
        H = np.vstack(hists)
        print(H.shape)
        H = H[:,np.where((H>0.001*16384).any(axis=0))[0]]
        print(H.shape)
        # back to list
        hists = [H[i] for i in range(H.shape[0])]

        # Prepare the parallelized list:
        n = len(hists)
        print(n)
        M_para = Parallel(n_jobs=16)(delayed(parallel_M)(h, hs, n) for h,hs in tqdm([(hists[i], hists[i:]) for i in range(n)]))
        M_para = np.vstack(M_para)


        M = M_para + M_para.T
        np.save(os.path.join(data_path, 'wass_matrices_hd', f'full_wass_dist_dategroup_{date_group}.npy'), M)
        print(f'Wasserstein matrix for Date Group {date_group} completed.')


if __name__ == "__main__":
    main()
