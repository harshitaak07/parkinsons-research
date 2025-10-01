import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def qc_filter(merged_df, exclude_col='exclude'):
    if exclude_col in merged_df.columns:
        merged_df = merged_df[merged_df[exclude_col] != True]
    return merged_df

def visualize_qc_metrics(df, qc_metrics=['snr', 'fd_mean', 'efc', 'fber']):
    for metric in qc_metrics:
        if metric in df.columns:
            sns.histplot(df[metric].dropna(), kde=True)
            plt.title(f'Distribution of {metric}')
            plt.xlabel(metric)
            plt.ylabel('Count')
            plt.show()

def handle_missing_data(df, missing_col_thresh=0.5, missing_row_thresh=0.3):
    df = df.loc[:, df.isnull().mean() < missing_col_thresh]
    df = df.loc[df.isnull().mean(axis=1) < missing_row_thresh, :]
    return df

def identify_feature_types(df, exclude_cols=['sub', 'ses']):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(exclude_cols)
    return list(numeric_cols), list(categorical_cols)

def visualize_missingness(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title('Missing Data Pattern')
    plt.show()

def remove_outliers(df, numeric_cols, z_thresh=5):
    z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std(ddof=0)
    mask = (z_scores.abs() > z_thresh).any(axis=1)
    return df.loc[~mask]
