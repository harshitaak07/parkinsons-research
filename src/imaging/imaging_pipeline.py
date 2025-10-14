# imaging_pipeline.py
import pandas as pd
import matplotlib.pyplot as plt
import os

files = {
    "DaTscan": "/mnt/data/DaTscan_Imaging_22Sep2025.csv",
    "DTI": "/mnt/data/DTI_Regions_of_Interest_22Sep2025.csv",
    "FS_CTH": "/mnt/data/FS7_APARC_CTH_22Sep2025.csv",
    "FS_SA": "/mnt/data/FS7_APARC_SA_22Sep2025.csv",
    "GreyMatter": "/mnt/data/Grey_Matter_Volume_22Sep2025.csv",
    "MRIQC": "/mnt/data/MRIQC_22Sep2025.csv",
    "Xing_SBR": "/mnt/data/Xing_Core_Lab_-_Quant_SBR_22Sep2025.csv",
    "Xing_Visual": "/mnt/data/Xing_Core_Lab_-_Visual_Read_22Sep2025.csv"
}

imaging_dfs = {name: pd.read_csv(path) for name, path in files.items()}

for name, df in imaging_dfs.items():
    if 'PATNO' in df.columns:
        df['PATNO'] = df['PATNO'].astype(str)
    if 'EVENT_ID' in df.columns:
        df['EVENT_ID'] = df['EVENT_ID'].astype(str)
    imaging_dfs[name] = df

def assign_visit_phase(df):
    df = df.copy()
    if 'PATNO' in df.columns:
        df['PATNO'] = df['PATNO'].astype(str)
    if 'EVENT_ID' in df.columns:
        df['EVENT_ID'] = df['EVENT_ID'].astype(str)
    visit_phase_mapping = {
        'BL':'early','SC':'early','R13':'early','R14':'mid','R15':'early',
        'V01':'mid','V02':'mid','V03':'mid','V04':'mid','V05':'mid','V06':'mid',
        'V07':'late','V08':'late','V09':'late','V10':'late','V11':'late','V12':'late',
        'V13':'late','V14':'late','V15':'late','R20':'mid','V19':'late','V20':'late','R21':'mid'
    }
    if 'EVENT_ID' in df.columns:
        df['visit_phase'] = df['EVENT_ID'].map(visit_phase_mapping).fillna('unknown')
    else:
        df['visit_phase'] = 'unknown'
    return df

imaging_dfs = {name: assign_visit_phase(df) for name, df in imaging_dfs.items()}

summary_stats = {}
for name, df in imaging_dfs.items():
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) == 0 or 'EVENT_ID' not in df.columns:
        continue
    visit_means = df.groupby('EVENT_ID')[num_cols].mean()
    summary_stats[name] = visit_means

visit_order = ['BL','SC','R13','R14','R15','V01','V02','V03','V04','V05','V06','V07','V08','V09','V10','V11','V12','V13','V14','V15','V19','V20']

def plot_visit_means(df_mean, title, max_features=10):
    df_mean = df_mean.loc[[v for v in visit_order if v in df_mean.index]]
    plt.figure(figsize=(12,6))
    for col in df_mean.columns[:max_features]:
        plt.plot(df_mean.index, df_mean[col], marker='o', linewidth=2, label=col)
    plt.title(title, fontsize=14)
    plt.xlabel('Visit (EVENT_ID)')
    plt.ylabel('Mean Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

for name, df_mean in summary_stats.items():
    plot_visit_means(df_mean, f"{name} Imaging Metric Progression")

os.makedirs("processed_imaging", exist_ok=True)
for name, df in imaging_dfs.items():
    out_path = os.path.join("processed_imaging", f"{name}_Processed.parquet")
    df.to_parquet(out_path, index=False)
