# -*- coding: utf-8 -*-
"""
opals_preprocess.py

Cleans and preprocesses Opals gait datasets:
- Handles missing values and outliers
- Clips outliers
- Adds engineered features
- Computes risk groups
- Saves processed datasets
"""

import os
import pandas as pd
import numpy as np


def preprocess_opals(raw_path, processed_path=None, save=True):
    """
    Preprocess Opals gait dataset:
    - Handle missing values
    - Clip outliers
    - Add engineered features
    - Assign risk categories
    - Save processed data if required
    """
    df = pd.read_csv(raw_path)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    def clip_outliers(df_in, cols):
        for col in cols:
            if col in df_in:
                Q1, Q3 = df_in[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                df_in[col] = df_in[col].clip(lower, upper)
        return df_in

    groups = {
        'Walking Speed and Cadence': ['SP_U', 'CAD_U'],
        'Arm Swing Amplitude and Variability': ['RA_AMP_U', 'LA_AMP_U', 'RA_STD_U', 'LA_STD_U'],
        'Symmetry and Asymmetry Measures': ['SYM_U', 'ASYM_IND_U'],
        'Stride and Step Timing and Regularity': ['STR_T_U', 'STR_CV_U', 'STEP_REG_U', 'STEP_SYM_U'],
        'Movement Smoothness / Jerk Measures': ['JERK_T_U', 'R_JERK_U', 'L_JERK_U'],
        'Functional Mobility (TUG) Test Metrics': [
            'TUG1_DUR', 'TUG1_STEP_NUM', 'TUG1_STRAIGHT_DUR',
            'TUG1_TURNS_DUR', 'TUG1_STEP_REG', 'TUG1_STEP_SYM'
        ]
    }

    all_features = [feat for feats in groups.values() for feat in feats]
    available_features = [f for f in all_features if f in df.columns]
    df = clip_outliers(df, available_features)

    if 'RA_AMP_U' in df.columns and 'LA_AMP_U' in df.columns:
        df['Arm_Amplitude_Diff'] = (df['RA_AMP_U'] - df['LA_AMP_U']).abs()
        groups['Arm Swing Amplitude and Variability'].append('Arm_Amplitude_Diff')

    thresholds_group = {
        'Walking Speed and Cadence': {'High Risk': 1.04, 'Moderate Risk': 1.19, 'invert': True},
        'Arm Swing Amplitude and Variability': {'High Risk': 15, 'Moderate Risk': 30, 'invert': False},
        'Symmetry and Asymmetry Measures': {'High Risk': 0.20, 'Moderate Risk': 0.15, 'invert': False},
        'Stride and Step Timing and Regularity': {'High Risk': 10, 'Moderate Risk': 7, 'invert': False},
        'Movement Smoothness / Jerk Measures': {'High Risk': 0.35, 'Moderate Risk': 0.20, 'invert': False},
        'Functional Mobility (TUG) Test Metrics': {'High Risk': 14.5, 'Moderate Risk': 12, 'invert': False},
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

    for group_name, features in groups.items():
        feat_list = [f for f in features if f in df.columns]
        if not feat_list:
            continue
        group_mean = df[feat_list].mean(axis=1)
        th = thresholds_group.get(group_name)
        if th:
            high, mod, inv = th['High Risk'], th['Moderate Risk'], th['invert']
        else:
            high, mod, inv = np.percentile(group_mean, 33), np.percentile(group_mean, 66), False
        df[f'{group_name}_risk'] = group_mean.apply(assign_risk, args=(high, mod, inv))

    if save:
        if processed_path is None:
            processed_path = os.path.join("data", "processed", "cleaned_gait_dataset_opals.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)

    return df
