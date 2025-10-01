import pandas as pd
from functools import reduce
import os
import warnings

def standardize_ids(df, sub_col='PATNO', ses_col='EVENTID'):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if sub_col not in df.columns:
        raise ValueError(f"Subject column '{sub_col}' not found in DataFrame.")
    if ses_col not in df.columns:
        raise ValueError(f"Session column '{ses_col}' not found in DataFrame.")
    df = df.rename(columns={sub_col: 'sub', ses_col: 'ses'})
    df['sub'] = df['sub'].astype(str).str.strip()
    df['ses'] = df['ses'].astype(str).str.strip()
    return df

def load_and_standardize(filepaths, id_cols_map):
    if not filepaths:
        raise ValueError("No filepaths provided.")
    if len(filepaths) != len(id_cols_map):
        raise ValueError("Length of filepaths and id_cols_map must match.")
    dfs = []
    for fp, cols in zip(filepaths, id_cols_map):
        if not os.path.exists(fp):
            raise FileNotFoundError(f"File not found: {fp}")
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            raise ValueError(f"Error reading file {fp}: {e}")
        if df.empty:
            warnings.warn(f"File {fp} is empty. Skipping.")
            continue
        df = standardize_ids(df, sub_col=cols.get('sub_col', 'PATNO'), ses_col=cols.get('ses_col', 'EVENTID'))
        dfs.append(df.drop_duplicates())
    if not dfs:
        raise ValueError("No valid DataFrames loaded.")
    return dfs

def merge_datasets(dfs):
    if not dfs:
        raise ValueError("No DataFrames to merge.")
    if len(dfs) == 1:
        return dfs[0]
    try:
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=['sub', 'ses'], how='outer'), dfs)
    except KeyError as e:
        raise ValueError(f"Merge failed due to missing key columns: {e}")
    return merged_df

def save_merged(merged_df, filepath='data/processed/merged_initial.csv'):
    if merged_df.empty:
        warnings.warn("DataFrame is empty. Saving anyway.")
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        merged_df.to_csv(filepath, index=False)
    except Exception as e:
        raise ValueError(f"Error saving to {filepath}: {e}")
