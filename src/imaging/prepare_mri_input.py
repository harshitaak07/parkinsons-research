import pandas as pd
import numpy as np

df = pd.read_csv('MRI_features.csv', low_memory=False)
df['PATNO'] = df['PATNO'].astype(str)
df['EVENTID'] = df['EVENTID'].astype(str)

# Select MRI feature columns (e.g., all columns starting with 'MRI_')
mri_cols = [c for c in df.columns if c.startswith('MRI_')]
df[mri_cols] = df[mri_cols].apply(pd.to_numeric, errors='coerce')

grouped = df.groupby(['PATNO', 'visitphase'])[mri_cols].mean().reset_index()
pivoted = grouped.pivot(index='PATNO', columns='visitphase', values=mri_cols)
visitphases = ['early', 'mid', 'late']
array = np.stack([
    pivoted.xs(phase, level=1, axis=1).values for phase in visitphases
], axis=1)
print(f"MRI array shape: {array.shape}")
np.save('autoencoderinput_mri.npy', array)
