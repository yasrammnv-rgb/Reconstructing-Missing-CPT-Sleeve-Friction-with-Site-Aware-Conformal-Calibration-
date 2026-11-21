"""Site-cluster bootstrap CIs for metrics, per-region and per-SBT-zone errors, example profile figure."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml import zone
import figs  # applies rcParams
cv = pd.read_pickle('cv_predictions.pkl')
cv['region'] = cv.region.replace({'Alaska/Canada': 'Other (AK/HI/CA)'})
models = ['const_Rf', 'ridge', 'rf', 'gbdt_noloc', 'gbdt']
zt = zone(cv.Ic.values)
for m in models:
    cv['se_' + m] = (cv['p_' + m] - cv.y) ** 2
    cv['ok_' + m] = (zone(cv['Ic_' + m].values) == zt).astype(float)
site_s = cv.groupby('site')[[f'se_{m}' for m in models] + [f'ok_{m}' for m in models]].agg(['sum']).droplevel(1, axis=1)
site_n = cv.groupby('site').size()
rng = np.random.default_rng(1); B = 1000
sites = site_s.index.values
boot = []
for b in range(B):
    s = rng.choice(sites, len(sites), replace=True)
    S = site_s.loc[s].sum(); n = site_n.loc[s].sum()
    boot.append({**{f'RMSE_{m}': np.sqrt(S[f'se_{m}'] / n) for m in models}, **{f'ACC_{m}': S[f'ok_{m}'] / n for m in models}})
boot = pd.DataFrame(boot)
rows = []
for m in models:
    rows.append(dict(model=m, RMSE=np.sqrt(cv[f'se_{m}'].mean()), RMSE_lo=boot[f'RMSE_{m}'].quantile(.025), RMSE_hi=boot[f'RMSE_{m}'].quantile(.975),
                     ACC=cv[f'ok_{m}'].mean(), ACC_lo=boot[f'ACC_{m}'].quantile(.025), ACC_hi=boot[f'ACC_{m}'].quantile(.975)))
T = pd.DataFrame(rows); print(T.round(3).to_string()); T.to_csv('metrics_ci.csv', index=False)
d = boot['RMSE_rf'] - boot['RMSE_gbdt']; print('RF-GBDT RMSE diff 95% CI', d.quantile([.025, .975]).values)
d = boot['RMSE_gbdt'] - boot['RMSE_gbdt_noloc']; print('GBDT coords effect CI', d.quantile([.025, .975]).values)
reg = cv.groupby('region').apply(lambda g: pd.Series({'n_soundings': g.ID.nunique(), 'RMSE_constRf': np.sqrt(g.se_const_Rf.mean()), 'RMSE_RF': np.sqrt(g.se_rf.mean()), 'RMSE_GBDT': np.sqrt(g.se_gbdt.mean()), 'SBTacc_RF': g.ok_rf.mean()}))
print(reg.round(3)); reg.to_csv('metrics_region.csv')
names = ['gravelly sand', 'sands', 'sand mixtures', 'silt mixtures', 'clays', 'organic']
zz = pd.Series(zt).map(lambda k: f'{7-k} {names[k]}')
zm = cv.groupby(zz.values).apply(lambda g: pd.Series({'share': len(g) / len(cv), 'RMSE_RF': np.sqrt(g.se_rf.mean()), 'acc_RF': g.ok_rf.mean(), 'acc_constRf': g.ok_const_Rf.mean()}))
print(zm.round(3)); zm.to_csv('metrics_zone.csv')

# Fig 4: example profiles (soundings near median RF error)
err = cv.groupby('ID').se_rf.mean().pipe(np.sqrt)
ids = [err.sort_values().index[int(len(err) * q)] for q in (0.25, 0.5, 0.75)]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 4.2), sharey=False)
for ax, i, q in zip(axs, ids, ('P25', 'P50', 'P75')):
    g = cv[cv.ID == i]
    ax.fill_betweenx(g.z, 10 ** g.lo * 1e3, 10 ** g.hi * 1e3, color='#2a78d6', alpha=0.18, lw=0, label='90% CQR interval')
    ax.plot(g.fs * 1e3, g.z, color='#0b0b0b', lw=1, label='Measured fs')
    ax.plot(10 ** g.p_rf * 1e3, g.z, color='#eb6834', lw=1.2, label='RF prediction')
    ax.set_xscale('log'); ax.invert_yaxis(); ax.set_xlabel('fs (kPa)')
    ax.set_title(f'CPT {i} ({q} error)', fontsize=8.5, loc='left')
axs[0].set_ylabel('Depth (m)'); fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig('fig4_profiles.png'); plt.close()
