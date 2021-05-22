# Script to group hyper parameter search in a single csv
import argparse
import numpy as np
import pandas as pd
import os

def main():
    """Iterate over all run results"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'ROOT',
        type=str,
        help='Root directory containing test tube results'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='output_filename'
    )
    args = parser.parse_args()

    # First check all versions in the folder
    versions = os.listdir(args.ROOT)

    print(f'There are {len(versions)} versions for the random search.')
    final_df = []
    for v in versions:
        # Load each version
        df = pd.read_csv(os.path.join(args.ROOT, v, 'meta_tags.csv'), 
                        index_col=0, skiprows=0).transpose()
        df['version']=v
        if 'test_r2' in df.columns:
            final_df.append(df)

    print(f'Only {len(final_df)} have finished training.')
    pd.concat(final_df).to_csv(args.output, index=False)

if __name__=='__main__':
    main()