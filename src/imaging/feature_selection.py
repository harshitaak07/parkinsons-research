from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import pandas as pd
import numpy as np
import category_encoders as ce
import warnings
import os

def build_preprocessing_pipeline(df, exclude_cols=['sub', 'ses']):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(exclude_cols, list):
        raise TypeError("exclude_cols must be a list.")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(exclude_cols)

    if len(numeric_cols) == 0 and len(categorical_cols) == 0:
        raise ValueError("No numeric or categorical columns found after excluding specified columns.")

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(('num', numeric_pipeline, numeric_cols))
    if len(categorical_cols) > 0:
        transformers.append(('cat', categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers)

    return preprocessor, list(numeric_cols), list(categorical_cols)

def fit_transform_and_save(df, preprocessor, exclude_cols=['sub', 'ses'], pipeline_path='models/preprocessing_pipeline.joblib', processed_csv_path='results/fused_features.csv'):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(exclude_cols, list):
        raise TypeError("exclude_cols must be a list.")
    missing_cols = [col for col in exclude_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Exclude columns not found in DataFrame: {missing_cols}")
    X = df.drop(columns=exclude_cols)
    if X.empty:
        raise ValueError("No columns left after dropping exclude columns.")
    try:
        X_processed = preprocessor.fit_transform(X)
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

    try:
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        joblib.dump(preprocessor, pipeline_path)
    except Exception as e:
        raise ValueError(f"Error saving pipeline to {pipeline_path}: {e}")

    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()

    processed_df = pd.DataFrame(X_processed)
    for col in exclude_cols:
        processed_df[col] = df[col].reset_index(drop=True)

    processed_df = processed_df[[c for c in processed_df.columns if c not in exclude_cols] + exclude_cols]

    try:
        os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
        processed_df.to_csv(processed_csv_path, index=False)
    except Exception as e:
        raise ValueError(f"Error saving processed data to {processed_csv_path}: {e}")

    return processed_df

def select_features(X, y, k=50):
    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples.")
    if X.shape[1] < k:
        warnings.warn(f"k ({k}) is greater than number of features ({X.shape[1]}). Setting k to {X.shape[1]}.")
        k = X.shape[1]
    if k <= 0:
        raise ValueError("k must be positive.")
    try:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
    except Exception as e:
        raise ValueError(f"Error during feature selection: {e}")
    return X_selected, selector

def target_encode(df, cat_cols, target):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(cat_cols, list):
        raise TypeError("cat_cols must be a list.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    missing_cols = [col for col in cat_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Categorical columns not found in DataFrame: {missing_cols}")
    try:
        encoder = ce.TargetEncoder(cols=cat_cols)
        df[cat_cols] = encoder.fit_transform(df[cat_cols], df[target])
    except Exception as e:
        raise ValueError(f"Error during target encoding: {e}")
    return df, encoder
