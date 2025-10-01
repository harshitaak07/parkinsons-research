import pandas as pd
from functools import reduce

def standardize_ids(df, sub_col='PATNO', ses_col='EVENTID'):
    df = df.rename(columns={sub_col: 'sub', ses_col: 'ses'})
    df['sub'] = df['sub'].astype(str).str.strip()
    df['ses'] = df['ses'].astype(str).str.strip()
    return df

def load_and_standardize(filepaths, id_cols_map):
    dfs = []
    for fp, cols in zip(filepaths, id_cols_map):
        df = pd.read_csv(fp)
        df = standardize_ids(df, sub_col=cols.get('sub_col', 'PATNO'), ses_col=cols.get('ses_col', 'EVENTID'))
        dfs.append(df.drop_duplicates())
    return dfs

def merge_datasets(dfs):
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub', 'ses'], how='outer'), dfs)
    return merged_df

def save_merged(merged_df, filepath='data/processed/merged_initial.csv'):
    merged_df.to_csv(filepath, index=False)
