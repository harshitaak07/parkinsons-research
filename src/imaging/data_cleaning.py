import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

def qc_filter(merged_df, exclude_col='exclude'):
    if merged_df.empty:
        warnings.warn("Input DataFrame is empty. Returning as is.")
        return merged_df
    if exclude_col in merged_df.columns:
        merged_df = merged_df[merged_df[exclude_col] != True]
    else:
        warnings.warn(f"Exclude column '{exclude_col}' not found. Skipping QC filter.")
    return merged_df

def visualize_qc_metrics(df, qc_metrics=['snr', 'fd_mean', 'efc', 'fber']):
    if df.empty:
        warnings.warn("DataFrame is empty. Skipping visualization.")
        return
    available_metrics = [metric for metric in qc_metrics if metric in df.columns]
    if not available_metrics:
        warnings.warn("None of the specified QC metrics found in DataFrame. Skipping visualization.")
        return
    for metric in available_metrics:
        data = df[metric].dropna()
        if data.empty:
            warnings.warn(f"No valid data for metric '{metric}'. Skipping.")
            continue
        sns.histplot(data, kde=True)
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Count')
        plt.show()

def handle_missing_data(df, missing_col_thresh=0.5, missing_row_thresh=0.3):
    if df.empty:
        warnings.warn("Input DataFrame is empty. Returning as is.")
        return df
    if not (0 <= missing_col_thresh <= 1):
        raise ValueError("missing_col_thresh must be between 0 and 1.")
    if not (0 <= missing_row_thresh <= 1):
        raise ValueError("missing_row_thresh must be between 0 and 1.")
    df = df.loc[:, df.isnull().mean() < missing_col_thresh]
    if df.empty:
        warnings.warn("All columns removed due to high missing data. Returning empty DataFrame.")
        return df
    df = df.loc[df.isnull().mean(axis=1) < missing_row_thresh, :]
    if df.empty:
        warnings.warn("All rows removed due to high missing data. Returning empty DataFrame.")
    return df

def identify_feature_types(df, exclude_cols=['sub', 'ses']):
    if df.empty:
        warnings.warn("DataFrame is empty. Returning empty lists.")
        return [], []
    if not isinstance(exclude_cols, list):
        raise TypeError("exclude_cols must be a list.")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(exclude_cols)
    return list(numeric_cols), list(categorical_cols)

def visualize_missingness(df):
    if df.empty:
        warnings.warn("DataFrame is empty. Skipping visualization.")
        return
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title('Missing Data Pattern')
    plt.show()

def remove_outliers(df, numeric_cols, z_thresh=5):
    if df.empty:
        warnings.warn("DataFrame is empty. Returning as is.")
        return df
    if not numeric_cols:
        warnings.warn("No numeric columns provided. Returning DataFrame as is.")
        return df
    available_cols = [col for col in numeric_cols if col in df.columns]
    if not available_cols:
        warnings.warn("None of the numeric columns found in DataFrame. Returning as is.")
        return df
    z_scores = (df[available_cols] - df[available_cols].mean()) / df[available_cols].std(ddof=0)
    # Handle columns with zero std
    z_scores = z_scores.replace([float('inf'), -float('inf')], 0)  # Replace inf with 0
    mask = (z_scores.abs() > z_thresh).any(axis=1)
    return df.loc[~mask]
