import os
import pandas as pd
pd.set_option('display.max_columns', None)
RAW_PATH = os.path.join("data", "raw", "Gait_Data___Arm_swing__Axivity__06Sep2025.csv")
df = pd.read_csv(RAW_PATH)
print(df.shape)
print(df.columns.tolist())
df.head()

identifier_cols = ['VISNO']
df = df.drop(columns=[col for col in identifier_cols if col in df.columns])
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
print(df.columns.tolist())

missing_summary = df.isnull().sum().sort_values(ascending=False)
print(missing_summary[missing_summary > 0])
df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
df = df.dropna(axis=0)
df.fillna(df.mean(numeric_only=True), inplace=True)

df['starttime'] = pd.to_datetime(df['starttime'], format='%m/%Y')
df['date'] = df['starttime']
df = df.sort_values(by=['patno', 'date'])
df.head()

data_quality_features = [
    'numberofdays', 'validdays6hr', 'validdays12hr',
    'nonweardetected', 'upsidedowndetected'
]
time_percent_features = [col for col in df.columns if 'percent' in col or 'time' in col]
svm_features = [col for col in df.columns if 'svm' in col]
step_bout_features = [col for col in df.columns if 'step' in col or 'bout' in col or 'nap' in col]
rotation_features = [col for col in df.columns if 'rotation' in col or 'velocityofrotation' in col]
gait_features = [col for col in df.columns if 'cadence' in col or 'rms' in col or 'amp' in col or 'stpreg' in col or 'stepasym' in col]
variability_features = [col for col in df.columns if 'cv' in col or 'sampentropy' in col]
std_features = [col for col in df.columns if col.endswith('std')]
feature_groups = {
    "data_quality": data_quality_features,
    "time_percent": time_percent_features,
    "svm": svm_features,
    "step_bout": step_bout_features,
    "rotation": rotation_features,
    "gait": gait_features,
    "variability": variability_features,
    "std": std_features,
}

import matplotlib.pyplot as plt
from IPython.display import display, HTML
sample_patients = df['patno'].dropna().unique()[:5]
for pat in sample_patients:
    sub_df = df[df['patno'] == pat]
    plt.figure()
    plt.plot(sub_df['date'], sub_df['stepcount'], marker='o')
    plt.title(f"Patient {pat} - Step Count Over Time")
    plt.xlabel("Date")
    plt.ylabel("Step Count")
    plt.grid(True)
    plt.show()
    display(HTML("<hr><br><br>"))

import seaborn as sns
from IPython.display import display, HTML
for group_name, features in feature_groups.items():
    for feat in features[:3]:
        plt.figure()
        sns.histplot(df[feat], kde=True, bins=30)
        plt.title(f"{feat} Distribution ({group_name})")
        plt.xlabel(feat)
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()
        display(HTML("<hr><br><br>"))

import numpy as np
selected_features = feature_groups["svm"] + feature_groups["gait"]
corr_matrix = df[selected_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap: SVM + Gait Features")
plt.show()

early_window = df[df['date'] < df['date'].min() + pd.Timedelta(days=30)]
low_step_patients = early_window.groupby('patno')['stepcount'].mean().sort_values().head(10)
print("Top 10 patients with lowest early step count:")
print(low_step_patients)

trend_df = df.groupby(pd.Grouper(key='date', freq='ME'))[['stepcount', 'meansvmdaymg']].mean()
plt.figure(figsize=(10, 6))
plt.plot(trend_df.index, trend_df['stepcount'], label='Step Count')
plt.plot(trend_df.index, trend_df['meansvmdaymg'], label='Mean SVM Day')
plt.title("Average Population Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_df = df.copy()
normalized_df[svm_features + gait_features + variability_features] = scaler.fit_transform(
    df[svm_features + gait_features + variability_features]
)

PROCESSED_PATH = os.path.join("data", "processed", "cleaned_gait_dataset_axivity.csv")
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)  # create folder if missing
df.to_csv(PROCESSED_PATH, index=False)

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

import matplotlib.pyplot as plt
plt.figure(figsize=(18, len(thresholds_group)*4))
plot_idx = 1
risk_summary = {}
for group_name, group_features in thresholds_group.items():
    feats = feature_groups[group_name]
    feats = [f for f in feats if f in df.columns]
    if not feats:
        continue
    group_mean = df[feats].mean(axis=1)
    high_thresh_val = np.percentile(group_mean, 100 * group_features['High Risk'])
    mod_thresh_val = np.percentile(group_mean, 100 * group_features['Moderate Risk'])
    invert = group_features['invert']
    risk_cat = group_mean.apply(assign_risk, args=(high_thresh_val, mod_thresh_val, invert))
    df[f'{group_name}_risk'] = risk_cat
    counts = risk_cat.value_counts().reindex(['Low Risk', 'Moderate Risk', 'High Risk']).fillna(0).astype(int)
    risk_summary[group_name] = counts
    plt.subplot(len(thresholds_group), 1, plot_idx)
    plt.hist(group_mean, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(high_thresh_val, color='red', linestyle='--', label='High Risk Threshold')
    plt.axvline(mod_thresh_val, color='orange', linestyle='--', label='Moderate Risk Threshold')
    plt.title(f"{group_name} - Risk Distribution")
    plt.xlabel('Group Mean Value')
    plt.ylabel('Patient Count')
    plt.legend()
    plot_idx += 1
plt.tight_layout()
plt.show()
print("\nPreliminary Risk Category Counts:")
for group_name, counts in risk_summary.items():
    print(f"\n{group_name.capitalize()} Risk:")
    print(counts)