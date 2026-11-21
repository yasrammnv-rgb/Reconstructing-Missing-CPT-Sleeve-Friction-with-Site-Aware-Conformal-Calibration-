import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300,
                     'font.family': 'DejaVu Sans'})
meta = pd.read_csv('meta.csv')
cv = pd.read_pickle('cv_predictions.pkl')
M = pd.read_csv('metrics.csv')

# Fig 1 map
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.scatter(meta.lon, meta.lat, s=8, c=C[0], edgecolor='white', linewidth=0.3)
ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)'); ax.set_aspect(1.2)
fig.tight_layout(); fig.savefig('fig1_map.png'); plt.close()

# Fig 2 model comparison
names = {'const_Rf': 'Constant Rf\n(empirical)', 'ridge': 'Ridge', 'gbdt_noloc': 'GBDT\n(no coords)',
         'gbdt': 'GBDT', 'rf': 'Random\nForest'}
M = M.set_index('model').loc[list(names)]
fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
for ax, col, lab in zip(axs, ['RMSE_log10fs', 'SBT_acc'], ['RMSE of log10 fs', 'SBT zone accuracy']):
    ax.bar(range(len(M)), M[col], color=C[0], width=0.6)
    for i, v in enumerate(M[col]):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(names.values(), fontsize=7.5)
    ax.set_title(lab, fontsize=9, color=INK, loc='left'); ax.grid(axis='x', visible=False)
axs[1].set_ylim(0.5, 0.8)
fig.tight_layout(); fig.savefig('fig2_models.png'); plt.close()

# Fig 3 obs vs pred (RF) hexbin
fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
for ax, k, t in zip(axs, ['p_const_Rf', 'p_rf'], ['Constant Rf baseline', 'Random Forest']):
    hb = ax.hexbin(cv.y, cv[k], gridsize=70, bins='log', cmap='Blues', mincnt=1, extent=(-3.3, 0.8, -3.3, 0.8))
    ax.plot([-3.3, 0.8], [-3.3, 0.8], color=INK2, lw=1)
    ax.set_xlabel('Measured log10 fs (MPa)'); ax.set_ylabel('Predicted log10 fs (MPa)'); ax.set_title(t, fontsize=9, loc='left')
fig.colorbar(hb, ax=axs, label='count (log)', shrink=0.8)
fig.savefig('fig3_obs_pred.png', bbox_inches='tight'); plt.close()

# Fig 5 importance
imp = pd.read_csv('perm_importance.csv', index_col=0).iloc[:, 0].head(10)[::-1]
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.barh(range(len(imp)), imp.values, color=C[0], height=0.6)
ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index); ax.set_xlabel('Permutation importance (Δ R²)')
ax.grid(axis='y', visible=False)
fig.tight_layout(); fig.savefig('fig5_importance.png'); plt.close()
print('ok')
