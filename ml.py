"""Leave-site-out (GroupKFold) benchmark: predict log10(fs) from qc-only features.
Models: empirical constant-Rf baseline, Ridge, Random Forest, histogram GBDT (with and without
coordinates). Uncertainty: conformalized quantile regression (CQR, Romano et al. 2019) with
GBDT quantile models, calibrated on held-out sites."""
import numpy as np, pandas as pd, json, time
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

PA = 0.1013
ALPHA = 0.10


def ic_from(qc, fs, sv, svp):
    qn = np.maximum(qc - sv, 1e-3)
    Fr = np.clip(fs / qn * 100, 0.01, 50)
    n = np.ones_like(qc)
    for _ in range(30):
        Qtn = (qn / PA) * (PA / svp) ** n
        Ic = np.sqrt((3.47 - np.log10(np.maximum(Qtn, 1e-3))) ** 2 + (np.log10(Fr) + 1.22) ** 2)
        n = np.minimum(0.381 * Ic + 0.05 * svp / PA - 0.15, 1.0)
    return Ic


def zone(ic):
    return np.digitize(ic, [1.31, 2.05, 2.6, 2.95, 3.6])


def hgb(**kw):
    p = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
             l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
             n_iter_no_change=30, random_state=0)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


if __name__ == '__main__':
    F = pd.read_pickle('features.pkl')
    F = F[F.fs > 5e-4].reset_index(drop=True)   # drop zero/unreliable sleeve readings (<0.5 kPa)
    feat_all = [c for c in F.columns if c not in ('ID', 'site', 'region', 'y', 'Ic', 'qc', 'fs', 'svp_true', 'sv_true')]
    feat_noloc = [c for c in feat_all if c not in ('lat', 'lon')]
    print(len(F), 'points', F.ID.nunique(), 'soundings', F.site.nunique(), 'sites', feat_all, flush=True)
    y = F.y.values; groups = F.site.values
    preds = {k: np.full(len(F), np.nan) for k in ['const_Rf', 'ridge', 'rf', 'gbdt_noloc', 'gbdt']}
    lo = np.full(len(F), np.nan); hi = np.full(len(F), np.nan)
    fold = np.zeros(len(F), int)
    imps = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(F, y, groups)):
        t0 = time.time(); fold[te] = k
        Ftr = F.iloc[tr]
        # empirical baseline: fs = qc * median(Rf) of training data
        preds['const_Rf'][te] = F.lqc.values[te] + np.median(y[tr] - F.lqc.values[tr])
        r = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(Ftr[feat_all], y[tr])
        preds['ridge'][te] = r.predict(F.iloc[te][feat_all])
        sub = np.random.default_rng(k).choice(tr, size=min(120000, len(tr)), replace=False)
        rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=10, max_features=0.4, n_jobs=2, random_state=0)
        rf.fit(F.iloc[sub][feat_all], y[sub])
        preds['rf'][te] = rf.predict(F.iloc[te][feat_all])
        preds['gbdt_noloc'][te] = hgb().fit(Ftr[feat_noloc], y[tr]).predict(F.iloc[te][feat_noloc])
        # GBDT + CQR: split training sites into proper-train / calibration
        gss = GroupShuffleSplit(1, test_size=0.2, random_state=k)
        ptr, cal = next(gss.split(Ftr, y[tr], groups[tr]))
        ptr, cal = tr[ptr], tr[cal]
        m = hgb().fit(F.iloc[tr][feat_all], y[tr])
        preds['gbdt'][te] = m.predict(F.iloc[te][feat_all])
        qlo = hgb(loss='quantile', quantile=ALPHA / 2).fit(F.iloc[ptr][feat_all], y[ptr])
        qhi = hgb(loss='quantile', quantile=1 - ALPHA / 2).fit(F.iloc[ptr][feat_all], y[ptr])
        Xc = F.iloc[cal][feat_all]
        s = np.maximum(qlo.predict(Xc) - y[cal], y[cal] - qhi.predict(Xc))
        # site-level calibration: use per-site max-score quantile would be conservative; use point scores
        q = np.quantile(s, np.ceil((len(s) + 1) * (1 - ALPHA)) / len(s))
        Xt = F.iloc[te][feat_all]
        lo[te] = qlo.predict(Xt) - q; hi[te] = qhi.predict(Xt) + q
        if k == 0:
            ss = np.random.default_rng(0).choice(te, 20000, replace=False)
            pi = permutation_importance(m, F.iloc[ss][feat_all], y[ss], n_repeats=5, random_state=0, n_jobs=1)
            imps = pd.Series(pi.importances_mean, index=feat_all).sort_values(ascending=False)
        print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)

    out = F[['ID', 'site', 'region', 'z', 'qc', 'fs', 'y', 'Ic', 'sv_true', 'svp_true']].copy()
    out['fold'] = fold
    for k, v in preds.items():
        out['p_' + k] = v
    out['lo'] = lo; out['hi'] = hi
    out.to_pickle('cv_predictions.pkl')
    imps.to_csv('perm_importance.csv')
    # metrics
    rows = []
    zt = zone(out.Ic.values)
    for k in preds:
        e = out['p_' + k] - out.y
        icp = ic_from(out.qc.values, 10 ** out['p_' + k].values, out.sv_true.values, out.svp_true.values)
        zp = zone(icp)
        rows.append(dict(model=k, RMSE_log10fs=np.sqrt(np.mean(e ** 2)), MAE_log10fs=np.mean(np.abs(e)),
                         R2=1 - np.mean(e ** 2) / np.var(out.y), MAE_Ic=np.mean(np.abs(icp - out.Ic)),
                         SBT_acc=np.mean(zp == zt), SBT_within1=np.mean(np.abs(zp - zt) <= 1)))
        out['Ic_' + k] = icp
    M = pd.DataFrame(rows); print(M.round(3).to_string())
    M.to_csv('metrics.csv', index=False)
    cov = ((out.y >= out.lo) & (out.y <= out.hi))
    print('CQR coverage', cov.mean(), 'mean width (log10)', (out.hi - out.lo).mean())
    print(cov.groupby(out.region).mean().round(3))
    out.to_pickle('cv_predictions.pkl')
    print(imps.head(12))
