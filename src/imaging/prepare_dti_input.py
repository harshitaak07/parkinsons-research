import pandas as pd
import numpy as np

df = pd.read_csv('DTI_features.csv', low_memory=False)
df['PATNO'] = df['PATNO'].astype(str)
df['EVENTID'] = df['EVENTID'].astype(str)

# Select DTI feature columns (e.g., all columns starting with 'DTI_')
dti_cols = [c for c in df.columns if c.startswith('DTI_')]
df[dti_cols] = df[dti_cols].apply(pd.to_numeric, errors='coerce')

grouped = df.groupby(['PATNO', 'visitphase'])[dti_cols].mean().reset_index()
pivoted = grouped.pivot(index='PATNO', columns='visitphase', values=dti_cols)
visitphases = ['early', 'mid', 'late']
array = np.stack([
    pivoted.xs(phase, level=1, axis=1).values for phase in visitphases
], axis=1)
print(f"DTI array shape: {array.shape}")
np.save('autoencoderinput_dti.npy', array)
