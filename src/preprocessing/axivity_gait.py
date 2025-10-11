# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_axivity(raw_path, processed_path=None, save=True):
    """
    Preprocess Axivity gait dataset:
    - Drop identifiers
    - Clean missing values
    - Normalize features
    - Assign risk categories
    - Save processed data if required
    """
    df = pd.read_csv(raw_path)
    # Keep VISNO for joining with clinical labels
    identifier_cols = []  # Don't drop VISNO anymore
    df = df.drop(columns=[col for col in identifier_cols if col in df.columns])
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))  
    df = df.dropna(axis=0)                             
    df.fillna(df.mean(numeric_only=True), inplace=True)

    if "starttime" in df.columns:
        df['starttime'] = pd.to_datetime(df['starttime'], format='%m/%Y', errors='coerce')
        df['date'] = df['starttime']
        df = df.sort_values(by=['patno', 'date'])

    feature_groups = {
        "svm": [col for col in df.columns if 'svm' in col],
        "gait": [col for col in df.columns if 'cadence' in col or 'rms' in col or 'amp' in col],
        "variability": [col for col in df.columns if 'cv' in col or 'sampentropy' in col],
        "step_bout": [col for col in df.columns if 'step' in col or 'bout' in col],
    }

    scaler = StandardScaler()
    for group in ["svm", "gait", "variability"]:
        feats = feature_groups[group]
        if feats:
            df[feats] = scaler.fit_transform(df[feats])

    thresholds_group = {
        'gait': {'High Risk': 0.33, 'Moderate Risk': 0.66, 'invert': False},
        'variability': {'High Risk': 0.33, 'Moderate Risk': 0.66, 'invert': False},
        'svm': {'High Risk': 0.33, 'Moderate Risk': 0.66, 'invert': True},
        'step_bout': {'High Risk': 0.33, 'Moderate Risk': 0.66, 'invert': False},
    }

    def assign_risk(value, high_thresh, mod_thresh, invert=False):
        if invert:
            if value > mod_thresh:
                return 'Low Risk'
            elif value > high_thresh:
                return 'Moderate Risk'
            else:
                return 'High Risk'
        else:
            if value < high_thresh:
                return 'Low Risk'
            elif value < mod_thresh:
                return 'Moderate Risk'
            else:
                return 'High Risk'

    for group_name, group_features in thresholds_group.items():
        feats = feature_groups.get(group_name, [])
        feats = [f for f in feats if f in df.columns]
        if not feats:
            continue
        group_mean = df[feats].mean(axis=1)
        high_thresh_val = np.percentile(group_mean, 100 * group_features['High Risk'])
        mod_thresh_val = np.percentile(group_mean, 100 * group_features['Moderate Risk'])
        df[f'{group_name}_risk'] = group_mean.apply(
            assign_risk, args=(high_thresh_val, mod_thresh_val, group_features['invert'])
        )

    if save:
        if processed_path is None:
            processed_path = os.path.join("data", "processed", "cleaned_gait_dataset_axivity.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)

    return df
