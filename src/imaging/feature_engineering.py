import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

class AsymmetryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, regions=None):
        self.regions = regions or ['caudate', 'putamen', 'thalamus']
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for region in self.regions:
            lh_col = f'lh_{region}'
            rh_col = f'rh_{region}'
            if lh_col in X.columns and rh_col in X.columns:
                denominator = X[lh_col] + X[rh_col]
                denominator = denominator.replace(0, np.nan)
                X[f'{region}_asym'] = (X[lh_col] - X[rh_col]) / denominator
            else:
                warnings.warn(f"Columns {lh_col} or {rh_col} not found. Skipping asymmetry for {region}.")
        return X

class ICVNormalizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, icv_col='ICV'):
        self.icv_col = icv_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        vol_cols = [col for col in X.columns if 'vol' in col.lower()]
        if self.icv_col in X.columns:
            for col in vol_cols:
                X[f'{col}_norm'] = X[col] / X[self.icv_col].replace(0, np.nan)
        else:
            warnings.warn(f"ICV column '{self.icv_col}' not found. Skipping normalization.")
        return X

class ROIAggregateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, roi_cols=None):
        self.roi_cols = roi_cols or ['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 'ROI6']
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        available_cols = [col for col in self.roi_cols if col in X.columns]
        if available_cols:
            X['roi_mean'] = X[available_cols].mean(axis=1)
            X['roi_max'] = X[available_cols].max(axis=1)
        else:
            warnings.warn("No ROI columns found. Skipping aggregation.")
        return X

class DaTscanAsymmetryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, regions=None):
        self.regions = regions or ['striatum', 'caudate', 'putamen']
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for region in self.regions:
            l_col = f'{region}_l_ref_cwm'
            r_col = f'{region}_r_ref_cwm'
            if l_col in X.columns and r_col in X.columns:
                denominator = X[l_col] + X[r_col]
                denominator = denominator.replace(0, np.nan)
                X[f'{region}_asym'] = (X[l_col] - X[r_col]) / denominator
            else:
                warnings.warn(f"Columns {l_col} or {r_col} not found. Skipping asymmetry for {region}.")
        return X

class GreyMatterNormalizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, gm_col='gm_volume', icv_col='icv'):
        self.gm_col = gm_col
        self.icv_col = icv_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if self.gm_col in X.columns and self.icv_col in X.columns:
            X[f'{self.gm_col}_norm'] = X[self.gm_col] / X[self.icv_col].replace(0, np.nan)
        else:
            warnings.warn(f"GM volume column '{self.gm_col}' or ICV column '{self.icv_col}' not found. Skipping normalization.")
        return X

class KeyMRIQCMetricsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key_metrics=None):
        self.key_metrics = key_metrics or ['snr_total', 'cjv', 'efc', 'fber']
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        available_metrics = [col for col in self.key_metrics if col in X.columns]
        if not available_metrics:
            warnings.warn("No key MRIQC metrics found.")
        return X

class VisualReadEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, vis_col='datscan_visintrp'):
        self.vis_col = vis_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if self.vis_col in X.columns and X[self.vis_col].dtype == 'object':
            X[f'{self.vis_col}_encoded'] = X[self.vis_col].astype('category').cat.codes
        return X
