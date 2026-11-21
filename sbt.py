"""Step 1: derive normalized CPT parameters (Robertson 2009; Robertson & Cabal 2010)
and group soundings into sites. qt is approximated by qc (no u2 in the database)."""
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from load import load

PA = 0.1013  # MPa
GW = 9.81e-3  # MN/m3


def sbt_profile(z, qc, fs, gwt):
    qt = np.maximum(qc, 1e-3)
    fs = np.maximum(fs, 1e-5)
    Rf = np.clip(fs / qt * 100, 0.05, 20)
    gam = (0.27 * np.log10(Rf) + 0.36 * np.log10(qt / PA) + 1.236) * GW
    gam = np.clip(gam, 14e-3, 22e-3)
    dz = np.diff(np.r_[0.0, z])
    sv = np.cumsum(gam * dz)
    u0 = GW * np.clip(z - gwt, 0, None)
    svp = np.maximum(sv - u0, 1e-3)
    qn = np.maximum(qt - sv, 1e-3)
    Fr = np.clip(fs / qn * 100, 0.01, 50)
    n = np.ones_like(qt)
    for _ in range(30):
        Qtn = (qn / PA) * (PA / svp) ** n
        Ic = np.sqrt((3.47 - np.log10(np.maximum(Qtn, 1e-3))) ** 2 + (np.log10(Fr) + 1.22) ** 2)
        n_new = np.minimum(0.381 * Ic + 0.05 * svp / PA - 0.15, 1.0)
        if np.max(np.abs(n_new - n)) < 1e-3:
            n = n_new; break
        n = n_new
    Qtn = (qn / PA) * (PA / svp) ** n
    Ic = np.sqrt((3.47 - np.log10(np.maximum(Qtn, 1e-3))) ** 2 + (np.log10(Fr) + 1.22) ** 2)
    return dict(svp=svp, sv=sv, Qtn=Qtn, Fr=Fr, Ic=Ic, gam=gam)


def region(lat, lon):
    if lat > 49 or lon < -125: return 'Alaska/Canada'
    if lon < -114: return 'Pacific'
    if lon < -100: return 'Mountain/Plains'
    if lat < 32.5 and lon < -80.5: return 'Gulf Coast'
    if lon >= -82 and lat < 37.5: return 'South Atlantic'
    if lon >= -80: return 'North Atlantic'
    return 'Central/Interior'


if __name__ == '__main__':
    meta, P = load()
    out = []
    for i, g in P.groupby('ID', sort=False):
        gwt = meta.loc[meta.ID == i, 'gwt'].iloc[0]
        r = sbt_profile(g.z.values, g.qc.values, g.fs.values, gwt)
        out.append(pd.DataFrame(r, index=g.index))
    P = P.join(pd.concat(out))
    # sites: soundings within 1 km
    X = np.radians(meta[['lat', 'lon']].values)
    meta['site'] = DBSCAN(eps=1.0 / 6371, min_samples=1, metric='haversine').fit_predict(X)
    meta['region'] = [region(a, b) for a, b in zip(meta.lat, meta.lon)]
    meta['Ic_mean'] = P.groupby('ID').Ic.mean().reindex(meta.ID).values
    print('sites', meta.site.nunique()); print(meta.region.value_counts())
    print(P[['Qtn', 'Fr', 'Ic', 'gam']].describe().T)
    zones = pd.cut(P.Ic, [0, 1.31, 2.05, 2.6, 2.95, 3.6, 9], labels=['7 gravelly sand', '6 sand', '5 sand mix', '4 silt mix', '3 clay', '2 organic'])
    print(zones.value_counts(normalize=True))
    meta.to_csv('meta.csv', index=False); P.to_pickle('profiles_sbt.pkl')
