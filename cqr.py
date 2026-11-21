"""Uncertainty comparison under leave-site-out CV:
(a) raw GBDT quantile regression, (b) marginal split-CQR, (c) Mondrian (region-conditional) CQR."""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from ml import hgb, ALPHA

F = pd.read_pickle('features.pkl')
F = F[F.fs > 5e-4].reset_index(drop=True)
feat = [c for c in F.columns if c not in ('ID', 'site', 'region', 'y', 'Ic', 'qc', 'fs', 'svp_true', 'sv_true')]
y = F.y.values; g = F.site.values; reg = F.region.values
lo_raw = np.full(len(F), np.nan); hi_raw = lo_raw.copy(); lo_m = lo_raw.copy(); hi_m = lo_raw.copy(); lo_r = lo_raw.copy(); hi_r = lo_raw.copy()


def qhat(s):
    n = len(s)
    return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n))


for k, (tr, te) in enumerate(GroupKFold(5).split(F, y, g)):
    ptr, cal = next(GroupShuffleSplit(1, test_size=0.25, random_state=k).split(tr, y[tr], g[tr]))
    ptr, cal = tr[ptr], tr[cal]
    ql = hgb(loss='quantile', quantile=ALPHA / 2).fit(F.iloc[ptr][feat], y[ptr])
    qh = hgb(loss='quantile', quantile=1 - ALPHA / 2).fit(F.iloc[ptr][feat], y[ptr])
    lc, hc = ql.predict(F.iloc[cal][feat]), qh.predict(F.iloc[cal][feat])
    lt, ht = ql.predict(F.iloc[te][feat]), qh.predict(F.iloc[te][feat])
    s = np.maximum(lc - y[cal], y[cal] - hc)
    lo_raw[te], hi_raw[te] = lt, ht
    q = qhat(s); lo_m[te], hi_m[te] = lt - q, ht + q
    for r in np.unique(reg[te]):
        mt = reg[te] == r; mc = reg[cal] == r
        qr = qhat(s[mc]) if mc.sum() >= 200 and len(np.unique(g[cal][mc])) >= 5 else q
        lo_r[te[mt]], hi_r[te[mt]] = lt[mt] - qr, ht[mt] + qr
    print('fold', k, flush=True)

out = F[['ID', 'site', 'region', 'z', 'y', 'Ic']].copy()
for nm, (a, b) in dict(raw=(lo_raw, hi_raw), cqr=(lo_m, hi_m), mondrian=(lo_r, hi_r)).items():
    out['lo_' + nm], out['hi_' + nm] = a, b
out.to_pickle('cqr_predictions.pkl')
rows = []
for nm in ['raw', 'cqr', 'mondrian']:
    c = (out.y >= out['lo_' + nm]) & (out.y <= out['hi_' + nm]); w = out['hi_' + nm] - out['lo_' + nm]
    d = dict(method=nm, coverage=c.mean(), width=w.mean())
    for r, v in c.groupby(out.region).mean().items():
        d['cov_' + r] = v
    # sounding-level: fraction of soundings whose coverage < 0.8
    d['frac_soundings_cov<0.8'] = (c.groupby(out.ID).mean() < 0.8).mean()
    rows.append(d)
T = pd.DataFrame(rows); T.to_csv('cqr_table.csv', index=False); print(T.T.round(3))
