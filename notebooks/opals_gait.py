# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
RAW_PATH = os.path.join("data", "raw", "Gait_Data___Arm_swing__Opals__07Aug2025.csv")
df = pd.read_csv(RAW_PATH)

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

def clip_outliers(df_in, cols):
    for col in cols:
        if col in df_in:
            Q1 = df_in[col].quantile(0.25)
            Q3 = df_in[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_in[col] = df_in[col].clip(lower, upper)
    return df_in

groups = {
    'Walking Speed and Cadence': ['SP_U', 'CAD_U'],
    'Arm Swing Amplitude and Variability': ['RA_AMP_U', 'LA_AMP_U', 'RA_STD_U', 'LA_STD_U'],
    'Symmetry and Asymmetry Measures': ['SYM_U', 'ASYM_IND_U'],
    'Stride and Step Timing and Regularity': ['STR_T_U', 'STR_CV_U', 'STEP_REG_U', 'STEP_SYM_U'],
    'Movement Smoothness / Jerk Measures': ['JERK_T_U', 'R_JERK_U', 'L_JERK_U'],
    'Functional Mobility (TUG) Test Metrics': ['TUG1_DUR', 'TUG1_STEP_NUM', 'TUG1_STRAIGHT_DUR', 'TUG1_TURNS_DUR', 'TUG1_STEP_REG', 'TUG1_STEP_SYM']
}

all_features = [feat for group_feats in groups.values() for feat in group_feats]
available_features = [f for f in all_features if f in df.columns]

df = clip_outliers(df, available_features)

if ('RA_AMP_U' in df.columns) and ('LA_AMP_U' in df.columns):
    df['Arm_Amplitude_Diff'] = (df['RA_AMP_U'] - df['LA_AMP_U']).abs()
    groups['Arm Swing Amplitude and Variability'].append('Arm_Amplitude_Diff')
    available_features.append('Arm_Amplitude_Diff')

thresholds_group = {
    'Walking Speed and Cadence': {'High Risk': 1.04, 'Moderate Risk': 1.19, 'invert': True},
    'Arm Swing Amplitude and Variability': {'High Risk': 15, 'Moderate Risk': 30, 'invert': False},
    'Symmetry and Asymmetry Measures': {'High Risk': 0.20, 'Moderate Risk': 0.15, 'invert': False},
    'Stride and Step Timing and Regularity': {'High Risk': 10, 'Moderate Risk': 7, 'invert': False},
    'Movement Smoothness / Jerk Measures': {'High Risk': 0.35, 'Moderate Risk': 0.20, 'invert': False},
    'Functional Mobility (TUG) Test Metrics': {'High Risk': 14.5, 'Moderate Risk': 12, 'invert': False}
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

plt.figure(figsize=(18, len(groups)*4))
plot_idx = 1
risk_summary = {}

for group_name, features in groups.items():
    feat_list = [f for f in features if f in available_features]
    if len(feat_list) == 0:
        continue

    group_mean = df[feat_list].mean(axis=1)

    if group_name in thresholds_group:
        high_thresh = thresholds_group[group_name]['High Risk']
        mod_thresh = thresholds_group[group_name]['Moderate Risk']
        invert = thresholds_group[group_name]['invert']
    else:
        high_thresh = np.percentile(group_mean, 33)
        mod_thresh = np.percentile(group_mean, 66)
        invert = False

    risk_cat = group_mean.apply(assign_risk, args=(high_thresh, mod_thresh, invert))
    counts = risk_cat.value_counts().reindex(['Low Risk', 'Moderate Risk', 'High Risk']).fillna(0).astype(int)
    risk_summary[group_name] = counts

    plt.subplot(len(groups), 1, plot_idx)
    plt.hist(group_mean, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    if invert:
        plt.axvline(mod_thresh, color='orange', linestyle='--', label='Moderate Risk Threshold')
        plt.axvline(high_thresh, color='red', linestyle='--', label='High Risk Threshold')
    else:
        plt.axvline(high_thresh, color='red', linestyle='--', label='High Risk Threshold')
        plt.axvline(mod_thresh, color='orange', linestyle='--', label='Moderate Risk Threshold')
    plt.title(f"{group_name} - Mean Feature Values Distribution")
    plt.xlabel('Feature Group Mean Value')
    plt.ylabel('Number of Patients')
    plt.legend()
    plt.grid(axis='y')
    plot_idx += 1

plt.tight_layout()
plt.show()

print("\nPreliminary Risk Category Counts Per Feature Group:")
for group_name, counts in risk_summary.items():
    print(f"\n{group_name}:")
    print(counts)

PROCESSED_PATH = os.path.join("data", "processed", "cleaned_gait_dataset_opals.csv")
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)