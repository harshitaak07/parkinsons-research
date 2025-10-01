from src.imaging.data_fusion import load_and_standardize, merge_datasets, save_merged
from src.imaging.data_cleaning import qc_filter, visualize_qc_metrics, handle_missing_data, identify_feature_types, visualize_missingness, remove_outliers
from src.imaging.feature_engineering import (
    AsymmetryTransformer, ICVNormalizationTransformer, ROIAggregateTransformer,
    DaTscanAsymmetryTransformer, GreyMatterNormalizationTransformer,
    KeyMRIQCMetricsSelector, VisualReadEncoder
)
from src.imaging.feature_selection import build_preprocessing_pipeline, fit_transform_and_save, select_features
from src.imaging.dimensionality_reduction import add_pca_embeddings, get_autoencoder_embeddings
from sklearn.pipeline import Pipeline
import json
import numpy as np
import pandas as pd

EMBEDDING_METHOD = 'pca'  # or 'autoencoder'
N_COMPONENTS = 20

filepaths = [
    'data/raw/imaging/DaTscan_Imaging_22Sep2025.csv',
    'data/raw/imaging/DTI_Regions_of_Interest_22Sep2025.csv',
    'data/raw/imaging/FS7_APARC_CTH_22Sep2025.csv',
    'data/raw/imaging/FS7_APARC_SA_22Sep2025.csv',
    'data/raw/imaging/Grey_Matter_Volume_22Sep2025.csv',
    'data/raw/imaging/MRIQC_22Sep2025.csv',
    'data/raw/imaging/Xing_Core_Lab_-_Quant_SBR_22Sep2025.csv',
    'data/raw/imaging/Xing_Core_Lab_-_Visual_Read_22Sep2025.csv'
]
id_cols_map = [
    {'sub_col': 'PATNO', 'ses_col': 'EVENT_ID'}, 
    {'sub_col': 'PATNO', 'ses_col': 'PAG_NAME'},  
    {'sub_col': 'PATNO', 'ses_col': 'EVENT_ID'},  
    {'sub_col': 'PATNO', 'ses_col': 'EVENT_ID'},  
    {'sub_col': 'PATNO', 'ses_col': 'EVENT_ID'},  
    {'sub_col': 'PATNO', 'ses_col': 'EVENT_ID'},  
    {'sub_col': 'PATNO', 'ses_col': 'EVENT_ID'},  
    {'sub_col': 'PATNO', 'ses_col': 'DATSCAN_DATE'}  
]
dfs = load_and_standardize(filepaths, id_cols_map)
merged_df = merge_datasets(dfs)
save_merged(merged_df)

merged_df = qc_filter(merged_df)
visualize_qc_metrics(merged_df)
merged_df = handle_missing_data(merged_df)
visualize_missingness(merged_df)
numeric_cols, categorical_cols = identify_feature_types(merged_df)
merged_df = remove_outliers(merged_df, numeric_cols)

feature_pipeline = Pipeline([
    ('asymmetry', AsymmetryTransformer(regions=['caudate', 'putamen', 'thalamus'])),
    ('icv_norm', ICVNormalizationTransformer(icv_col='ICV')),
    ('roi_agg', ROIAggregateTransformer(roi_cols=['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 'ROI6'])),
    ('datscan_asym', DaTscanAsymmetryTransformer(regions=['striatum', 'caudate', 'putamen'])),
    ('gm_norm', GreyMatterNormalizationTransformer(gm_col='gm_volume', icv_col='icv')),
    ('mriqc_select', KeyMRIQCMetricsSelector(key_metrics=['snr_total', 'cjv', 'efc', 'fber'])),
    ('visual_read', VisualReadEncoder(vis_col='datscan_visintrp')),
])
merged_df = feature_pipeline.fit_transform(merged_df)

preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline(merged_df)
processed_df = fit_transform_and_save(merged_df, preprocessor)
y = ...  # Define your target variable here (e.g., processed_df['diagnosis'].values)
X = processed_df.drop(columns=['sub', 'ses']).values
X_selected, selector = select_features(X, y, k=50)
selected_indices = selector.get_support(indices=True)
selected_feature_names = processed_df.drop(columns=['sub', 'ses']).columns[selected_indices]
selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)
selected_df['sub'] = processed_df['sub'].reset_index(drop=True)
selected_df['ses'] = processed_df['ses'].reset_index(drop=True)

if EMBEDDING_METHOD == 'pca':
    selected_df, pca_model = add_pca_embeddings(selected_df, list(selected_feature_names), n_components=N_COMPONENTS)
elif EMBEDDING_METHOD == 'autoencoder':
    embeddings = get_autoencoder_embeddings(selected_df.drop(columns=['sub', 'ses']).values, n_components=N_COMPONENTS, epochs=50, batch_size=32)
    for i in range(embeddings.shape[1]):
        selected_df[f'ae_{i+1}'] = embeddings[:, i]
else:
    raise ValueError('Unknown embedding method')
selected_df.to_csv('results/fused_features_with_embeddings.csv', index=False)

manifest_info = {
    'num_rows_after_filtering': selected_df.shape[0],
    'num_columns_after_filtering': selected_df.shape[1],
    'selected_features': list(selected_feature_names),
    'embedding_method': EMBEDDING_METHOD,
    'processed_data_path': 'results/fused_features_with_embeddings.csv'
}
with open('results/pipeline_manifest.json', 'w') as f:
    json.dump(manifest_info, f, indent=2)