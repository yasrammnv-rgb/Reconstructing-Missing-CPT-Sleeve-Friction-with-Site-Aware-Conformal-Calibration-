"""Additional analyses for the extended manuscript."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml import zone
import figs  # rcParams

cv = pd.read_pickle('cv_predictions.pkl')
cq = pd.read_pickle('cqr_predictions.pkl')
zt = zone(cv.Ic.values); zp = zone(cv.Ic_rf.values); zb = zone(cv.Ic_const_Rf.values)
names = ['7 Gravelly sand', '6 Sands', '5 Sand mix.', '4 Silt mix.', '3 Clays', '2 Organic']
# confusion matrix (row-normalized, %)
cm = pd.crosstab(pd.Series(zt, name='ref'), pd.Series(zp, name='pred'), normalize='index') * 100
cm = cm.reindex(index=range(6), columns=range(6), fill_value=0)
cm.index = names; cm.columns = names
cm.round(1).to_csv('confusion_rf.csv'); print(cm.round(1))
# depth bins
bins = [0, 2, 5, 10, 20, 70]
db = pd.cut(cv.z, bins)
dep = cv.groupby(db, observed=True).apply(lambda g: pd.Series({
    'n_points': len(g), 'RMSE_constRf': np.sqrt(((g.p_const_Rf - g.y) ** 2).mean()),
    'RMSE_RF': np.sqrt(((g.p_rf - g.y) ** 2).mean()),
    'SBTacc_RF': (zone(g.Ic_rf.values) == zone(g.Ic.values)).mean()}))
print(dep.round(3)); dep.to_csv('metrics_depth.csv')
# sounding-level
def r2(g):
    v = np.var(g.y); return 1 - np.mean((g.p_rf - g.y) ** 2) / v if v > 0 else np.nan
snd = cv.groupby('ID').apply(lambda g: pd.Series({'RMSE_RF': np.sqrt(((g.p_rf - g.y) ** 2).mean()),
                                                  'RMSE_const': np.sqrt(((g.p_const_Rf - g.y) ** 2).mean()),
                                                  'R2_RF': r2(g), 'acc_RF': (zone(g.Ic_rf.values) == zone(g.Ic.values)).mean()}))
s = dict(median_RMSE_RF=snd.RMSE_RF.median(), IQR_RMSE_RF=(snd.RMSE_RF.quantile(.25), snd.RMSE_RF.quantile(.75)),
         frac_RF_better=(snd.RMSE_RF < snd.RMSE_const).mean(), median_acc=snd.acc_RF.median(),
         frac_acc_gt_0_8=(snd.acc_RF > 0.8).mean(), frac_acc_lt_0_5=(snd.acc_RF < 0.5).mean(),
         median_R2=snd.R2_RF.median())
print(s); pd.Series(s).to_csv('metrics_sounding.csv')
# CQR coverage & width by SBT zone
cq['zone'] = [names[k] for k in zone(cq.Ic.values)]
cov = cq.groupby('zone').apply(lambda g: pd.Series({
    'cov_raw': ((g.y >= g.lo_raw) & (g.y <= g.hi_raw)).mean(),
    'cov_cqr': ((g.y >= g.lo_cqr) & (g.y <= g.hi_cqr)).mean(),
    'width_cqr': (g.hi_cqr - g.lo_cqr).mean()})).reindex(names)
print(cov.round(3)); cov.to_csv('cqr_by_zone.csv')

# Fig 7: confusion matrix heatmap
fig, ax = plt.subplots(figsize=(4.8, 4.0))
im = ax.imshow(cm.values, cmap='Blues', vmin=0, vmax=100)
for i in range(6):
    for j in range(6):
        v = cm.values[i, j]
        ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7.5, color='white' if v > 55 else '#0b0b0b')
ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7.5)
ax.set_yticks(range(6)); ax.set_yticklabels(names, fontsize=7.5)
ax.set_xlabel('Predicted zone (RF)'); ax.set_ylabel('Reference zone'); ax.grid(False)
fig.colorbar(im, ax=ax, label='% of reference zone', shrink=0.8)
fig.tight_layout(); fig.savefig('fig7_confusion.png'); plt.close()

# Fig 8: sounding-level RMSE distribution, RF vs constant Rf
fig, ax = plt.subplots(figsize=(4.8, 3.0))
b = np.linspace(0, 1.2, 41)
ax.hist(snd.RMSE_const, bins=b, color='#eb6834', alpha=0.85, label='Constant Rf')
ax.hist(snd.RMSE_RF, bins=b, color='#2a78d6', alpha=0.85, label='Random forest')
ax.set_xlabel('Sounding-level RMSE of log10 fs'); ax.set_ylabel('Number of soundings')
ax.legend(frameon=False, fontsize=8); ax.grid(axis='x', visible=False)
fig.tight_layout(); fig.savefig('fig8_sounding_rmse.png'); plt.close()
