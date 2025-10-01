from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import pandas as pd
import numpy as np
import category_encoders as ce

def build_preprocessing_pipeline(df, exclude_cols=['sub', 'ses']):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(exclude_cols)

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return preprocessor, list(numeric_cols), list(categorical_cols)

def fit_transform_and_save(df, preprocessor, exclude_cols=['sub', 'ses'], pipeline_path='models/preprocessing_pipeline.joblib', processed_csv_path='results/fused_features.csv'):
    X = df.drop(columns=exclude_cols)
    X_processed = preprocessor.fit_transform(X)

    joblib.dump(preprocessor, pipeline_path)

    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()

    processed_df = pd.DataFrame(X_processed)
    for col in exclude_cols:
        processed_df[col] = df[col].reset_index(drop=True)

    processed_df = processed_df[[c for c in processed_df.columns if c not in exclude_cols] + exclude_cols]

    processed_df.to_csv(processed_csv_path, index=False)

    return processed_df

def select_features(X, y, k=50):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector

def target_encode(df, cat_cols, target):
    encoder = ce.TargetEncoder(cols=cat_cols)
    df[cat_cols] = encoder.fit_transform(df[cat_cols], df[target])
    return df, encoder
