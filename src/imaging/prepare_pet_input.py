import pandas as pd
import numpy as np

df = pd.read_csv('PET_features.csv', low_memory=False)
df['PATNO'] = df['PATNO'].astype(str)
df['EVENTID'] = df['EVENTID'].astype(str)

# Select PET feature columns (e.g., all columns starting with 'PET_')
pet_cols = [c for c in df.columns if c.startswith('PET_')]
df[pet_cols] = df[pet_cols].apply(pd.to_numeric, errors='coerce')

grouped = df.groupby(['PATNO', 'visitphase'])[pet_cols].mean().reset_index()
pivoted = grouped.pivot(index='PATNO', columns='visitphase', values=pet_cols)
visitphases = ['early', 'mid', 'late']
array = np.stack([
    pivoted.xs(phase, level=1, axis=1).values for phase in visitphases
], axis=1)
print(f"PET array shape: {array.shape}")
np.save('autoencoderinput_pet.npy', array)
