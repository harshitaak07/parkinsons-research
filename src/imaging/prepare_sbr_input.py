import pandas as pd
import numpy as np

df = pd.read_csv('Xing_Core_Lab_-_Quant_SBR_22Sep2025.csv', low_memory=False)
df['PATNO'] = df['PATNO'].astype(str)
df['EVENTID'] = df['EVENTID'].astype(str)

# Select SBR columns
sbr_cols = [c for c in df.columns if c.endswith('REFCWM')]
df[sbr_cols] = df[sbr_cols].apply(pd.to_numeric, errors='coerce')

# Assume 'visitphase' column exists
grouped = df.groupby(['PATNO', 'visitphase'])[sbr_cols].mean().reset_index()
pivoted = grouped.pivot(index='PATNO', columns='visitphase', values=sbr_cols)
visitphases = ['early', 'mid', 'late']
array = np.stack([
    pivoted.xs(phase, level=1, axis=1).values for phase in visitphases
], axis=1)
print(f"SBR array shape: {array.shape}")
np.save('autoencoderinput_sbr.npy', array)
