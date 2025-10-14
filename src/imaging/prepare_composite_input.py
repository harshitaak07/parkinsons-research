import pandas as pd
import numpy as np

df = pd.read_csv('Composite_imaging_features.csv', low_memory=False)
df['PATNO'] = df['PATNO'].astype(str)
df['EVENTID'] = df['EVENTID'].astype(str)

# Select composite feature columns (customize as needed)
composite_cols = [c for c in df.columns if c.startswith('IMG_')]
df[composite_cols] = df[composite_cols].apply(pd.to_numeric, errors='coerce')

grouped = df.groupby(['PATNO', 'visitphase'])[composite_cols].mean().reset_index()
pivoted = grouped.pivot(index='PATNO', columns='visitphase', values=composite_cols)
visitphases = ['early', 'mid', 'late']
array = np.stack([
    pivoted.xs(phase, level=1, axis=1).values for phase in visitphases
], axis=1)
print(f"Composite array shape: {array.shape}")
np.save('autoencoderinput_composite.npy', array)
