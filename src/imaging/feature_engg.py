def create_asymmetry_features(df, regions=['caudate', 'putamen', 'thalamus']):
    for region in regions:
        lh_col = f'lh_{region}'
        rh_col = f'rh_{region}'
        if lh_col in df.columns and rh_col in df.columns:
            denominator = df[lh_col] + df[rh_col]
            denominator = denominator.replace(0, pd.NA)
            df[f'{region}_asym'] = (df[lh_col] - df[rh_col]) / denominator
    return df

def normalize_volumes_by_icv(df, numeric_cols, icv_col='ICV'):
    if icv_col in df.columns:
        vol_cols = [col for col in numeric_cols if 'vol' in col.lower()]
        for col in vol_cols:
            # Avoid division by zero
            df[f'{col}_norm'] = df[col] / df[icv_col].replace(0, pd.NA)
    return df

def add_roi_aggregates(df, roi_cols):
    if roi_cols:
        df['roi_mean'] = df[roi_cols].mean(axis=1)
        df['roi_max'] = df[roi_cols].max(axis=1)
    return df
