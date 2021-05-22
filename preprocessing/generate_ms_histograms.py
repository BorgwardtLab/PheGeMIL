import numpy as np
import pandas as pd
import os
from tqdm import tqdm


# Read and save histograms for MS images
def main():
    data_path = '/links/groups/borgwardt/Data/Jesse_2018'
    df_mil = pd.read_csv(os.path.join(data_path,
                                    'csv',
                                    'df_20200121_numpy_MIL_npy_coordinates_dates_genofiltered_dems.csv'))

    df_mil.head()

    # Extract the histograms for groups of dates
    dates_1 = ['0118', '0119'] # 2 days span
    dates_2 = ['0223', '0226', '0301', '0302'] # 7 days span
    dates_3 = ['0305', '0307', '0309'] # 4 days span
    dates_4 = ['0313', '0315', '0319', '0320', '0321'] # 8 days span

    group_dict = {'0118': 1, '0119': 1, 
                '0223': 2, '0226': 2, '0301': 2, '0302': 2,
                '0305': 3, '0307': 3, '0309': 3,
                '0313': 4, '0315': 4, '0319': 4, '0320': 4, '0321':4}
    group_dict_enc = {0: 1, 1: 1, 
                2: 2, 3: 2, 4: 2, 5: 2,
                6: 3, 7: 3, 8: 3, 9: 3,
                10: 4, 11: 4, 12: 4, 13:4, 14:4, 15:4}

    def get_histogram(img, lims=[np.array([0,0.2,0.4,0.6,0.8,1]),
                            np.array([0,0.2,0.4,0.6,0.8,1]),
                            np.array([0,0.2,0.4,0.6,0.8,1]),
                            np.array([0,0.6,1.2,1.8,2.4,3]),
                            np.array([0,0.3,0.6,0.9,1.2,1.5])], normalize=True):
        # Normalize by the number of images in the block
        hist = np.histogramdd(np.moveaxis(img, 0, -1).reshape(5,-1).T, lims)[0]
        if normalize:
            return hist/img.shape[0]
        else:
            return hist
    def get_dated_images(img, dates, date_group):
        date_idx = np.array([group_dict_enc[i] for i in dates])
        if date_group == 'all':
            idx = date_idx.astype(bool)
        else:
            idx = date_idx == date_group
        return img[idx]

    df_mil['dates_enc'] = df_mil.dates_enc.apply(lambda x: [int(i) for i in x[1:-1].split()])

    idx = [group_dict_enc[i] for i in df_mil.iloc[0]['dates_enc']]

    for i, row in tqdm(df_mil.iterrows()):
        # load imgs 
        img = np.load(os.path.join(data_path, 'numpy_MIL_resized', '2017-2018_CIMMYT_Wheat', 
                                row['Path'], row['PlotID']+'.npy'))
#         for date_group in [1,2,3,4,'all']:
        for date_group in [1,3,4,'all']:
            img_date = get_dated_images(img, row['dates_enc'], date_group)
    #         print(img_date.shape)
#             h = get_histogram(img_date)
            h = get_histogram(img_date, lims=[np.array([x/10 for x in range(0,11)]),
                           np.array([x/10 for x in range(0,11)]),
                           np.array([x/10 for x in range(0,11)]),
                           np.array([0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3]),
                           np.array([0,0.15,0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.35,1.5])])
            # Save histogram
            np.save(os.path.join(data_path, 'numpy_MIL_resized', 'histograms_ms_hd', f'date_group_{date_group}',
                                row['PlotID']+'.npy'), h)

if __name__ == "__main__":
    main()