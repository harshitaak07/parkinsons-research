import pandas as pd
import matplotlib.pyplot as plt

sbr = pd.read_csv('Xing_Core_Lab_-_Quant_SBR_22Sep2025.csv', low_memory=False)

sbr['PATNO'] = sbr['PATNO'].astype(str)
sbr['EVENTID'] = sbr['EVENTID'].astype(str)

sbr_cols = [c for c in sbr.columns if c.endswith('REFCWM')]

for c in sbr_cols:
    sbr[c] = pd.to_numeric(sbr[c], errors='coerce')

visitorder = sorted(sbr['EVENTID'].unique())
means = sbr.groupby('EVENTID')[sbr_cols].mean().loc[visitorder]
peaks = means.idxmax()      
peakvalues = means.max()    

summary = pd.DataFrame({
    'Region': sbr_cols,
    'Peak Visit': peaks.values,
    'Peak Value': peakvalues.values
})
print("Peak summary:")
print(summary)

visitseq = sbr.groupby('PATNO')['EVENTID'].apply(list)
positions = {}
for visits in visitseq:
    for pos, eid in enumerate(visits, 1):
        if eid not in positions:
            positions[eid] = []
        positions[eid].append(pos)
meanpositions = {eid: sum(poslist)/len(poslist) for eid, poslist in positions.items()}
sortedeids = sorted(meanpositions, key=meanpositions.get)
n = len(sortedeids)
visitcategories = {}
for i, eid in enumerate(sortedeids):
    if i < n//3:
        visitcategories[eid] = 'early'
    elif i < 2*n//3:
        visitcategories[eid] = 'mid'
    else:
        visitcategories[eid] = 'late'
sbr['visitphase'] = sbr['EVENTID'].map(visitcategories)

print("Visit phase mapping:")
print(sbr[['EVENTID', 'visitphase']].drop_duplicates())


region = sbr_cols[0]
plt.figure(figsize=(10,6))
for phase, group in sbr.groupby('visitphase'):
    mean_vals = group.groupby('EVENTID')[region].mean()
    plt.plot(mean_vals.index, mean_vals.values, label=phase)
plt.xlabel('EVENTID')
plt.ylabel(f'Mean {region}')
plt.title(f'Mean {region} SBR over Visit Phases')
plt.legend()
plt.show()
