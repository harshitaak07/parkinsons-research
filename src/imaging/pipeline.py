import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.imaging.data_fusion import load_and_standardize, merge_datasets, save_merged
from src.imaging.data_cleaning import (
    qc_filter, visualize_qc_metrics, handle_missing_data,
    identify_feature_types, visualize_missingness, remove_outliers
)
from src.imaging.feature_engineering import (
    AsymmetryTransformer, ICVNormalizationTransformer, ROIAggregateTransformer,
    DaTscanAsymmetryTransformer, GreyMatterNormalizationTransformer,
    KeyMRIQCMetricsSelector, VisualReadEncoder
)
from src.imaging.feature_selection import build_preprocessing_pipeline, fit_transform_and_save, select_features
from src.imaging.dimensionality_reduction import add_pca_embeddings, get_autoencoder_embeddings

EMBEDDING_METHOD = 'autoencoder'        # or 'autoencoder'
N_COMPONENTS = 20
TARGET_COLUMN = 'diagnosis'     # <-- replace with your real target if applicable
RAW_DIR = 'data/raw/imaging'
RESULTS_DIR = 'results'
PROCESSED_DIR = 'data/processed'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

filepaths = [
    f'{RAW_DIR}/DaTscan_Imaging_22Sep2025.csv',
    f'{RAW_DIR}/DTI_Regions_of_Interest_22Sep2025.csv',
    f'{RAW_DIR}/FS7_APARC_CTH_22Sep2025.csv',
    f'{RAW_DIR}/FS7_APARC_SA_22Sep2025.csv',
    f'{RAW_DIR}/Grey_Matter_Volume_22Sep2025.csv',
    f'{RAW_DIR}/MRIQC_22Sep2025.csv',
    f'{RAW_DIR}/Xing_Core_Lab_-_Quant_SBR_22Sep2025.csv',
    f'{RAW_DIR}/Xing_Core_Lab_-_Visual_Read_22Sep2025.csv'
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


print("Loading and merging data...")
dfs = load_and_standardize(filepaths, id_cols_map)
merged_df = merge_datasets(dfs)
save_merged(merged_df,f'{PROCESSED_DIR}/merged_initial.csv')
print("Merged data shape:", merged_df.shape)
print("\nRunning quality control and cleaning...")
merged_df = qc_filter(merged_df)
visualize_qc_metrics(merged_df)
merged_df = handle_missing_data(merged_df)
visualize_missingness(merged_df)
numeric_cols, categorical_cols = identify_feature_types(merged_df)
merged_df = remove_outliers(merged_df, numeric_cols)
merged_df.to_csv(f'{PROCESSED_DIR}/merged_cleaned.csv', index=False)
print("Cleaned data shape:", merged_df.shape)


print("\nApplying feature engineering pipeline...")
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
merged_df.to_csv(f'{PROCESSED_DIR}/features_engineered.csv', index=False)
print("Feature-engineered data shape:", merged_df.shape)
print("\nBuilding preprocessing pipeline...")
preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline(merged_df)
processed_df = fit_transform_and_save(
    merged_df,
    preprocessor,
    exclude_cols=['sub', 'ses'],
    pipeline_path=f'{PROCESSED_DIR}/preprocessing_pipeline.joblib',
    processed_csv_path=f'{PROCESSED_DIR}/fused_features.csv'
)

print("\nSelecting top features...")
if TARGET_COLUMN in processed_df.columns:
    y = processed_df[TARGET_COLUMN].values
    X = processed_df.drop(columns=['sub', 'ses', TARGET_COLUMN]).values
else:
    print(f"Warning: Target column '{TARGET_COLUMN}' not found. Using dummy y.")
    y = np.zeros(processed_df.shape[0])
    X = processed_df.drop(columns=['sub', 'ses']).values
X_selected, selector = select_features(X, y, k=50)
selected_indices = selector.get_support(indices=True)
selected_feature_names = processed_df.drop(columns=['sub', 'ses']).columns[selected_indices]
selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)
selected_df['sub'] = processed_df['sub'].reset_index(drop=True)
selected_df['ses'] = processed_df['ses'].reset_index(drop=True)
selected_df.to_csv(f'{PROCESSED_DIR}/selected_features.csv', index=False)
print("Selected data shape:", selected_df.shape)


print(f"\nGenerating {EMBEDDING_METHOD.upper()} embeddings...")
if EMBEDDING_METHOD == 'pca':
    selected_df, pca_model = add_pca_embeddings(
        selected_df, list(selected_feature_names), n_components=N_COMPONENTS
    )
elif EMBEDDING_METHOD == 'autoencoder':
    X_in = selected_df.drop(columns=['sub', 'ses']).values
    embeddings = get_autoencoder_embeddings(
        X_in, n_components=N_COMPONENTS, epochs=50, batch_size=32
    )
    for i in range(embeddings.shape[1]):
        selected_df[f'ae_{i+1}'] = embeddings[:, i]
else:
    raise ValueError("Unknown EMBEDDING_METHOD. Use 'pca' or 'autoencoder'.")
selected_df.to_csv(f'{RESULTS_DIR}/fused_features_with_embeddings.csv', index=False)
print("Final data with embeddings shape:", selected_df.shape)


manifest_info = {
    'num_rows_after_filtering': int(selected_df.shape[0]),
    'num_columns_after_filtering': int(selected_df.shape[1]),
    'selected_features': list(selected_feature_names),
    'embedding_method': EMBEDDING_METHOD,
    'processed_data_path': f'{RESULTS_DIR}/fused_features_with_embeddings.csv'
}
with open(f'{RESULTS_DIR}/pipeline_manifest.json', 'w') as f:
    json.dump(manifest_info, f, indent=2)
print("\nPipeline completed successfully!")
print(f"Results saved to: {RESULTS_DIR}/fused_features_with_embeddings.csv")