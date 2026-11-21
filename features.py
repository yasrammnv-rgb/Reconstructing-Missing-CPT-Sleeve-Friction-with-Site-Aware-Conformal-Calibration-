"""Build point-wise feature table using ONLY qc-derived and metadata features.
Target: log10(fs). Resampled to 0.05 m."""
import numpy as np, pandas as pd

DZ = 0.05
WINS = [0.25, 0.5, 1.0, 2.0]  # m


def build():
    P = pd.read_pickle('profiles_sbt.pkl')
    meta = pd.read_csv('meta.csv').set_index('ID')
    rows = []
    step = int(round(DZ / 0.01))
    for i, g in P.groupby('ID', sort=False):
        lq = pd.Series(np.log10(np.maximum(g.qc.values, 1e-3)))
        f = pd.DataFrame(index=g.index)
        f['ID'] = i
        f['z'] = g.z.values
        f['lqc'] = lq.values
        m = meta.loc[i]
        f['z_gwt'] = g.z.values - m.gwt
        f['gwt'] = m.gwt
        # stress from a qc-only unit weight proxy (no fs used): constant 18 kN/m3
        sv = 0.018 * g.z.values
        svp = np.maximum(sv - 9.81e-3 * np.clip(g.z.values - m.gwt, 0, None), 1e-3)
        f['lsvp'] = np.log10(svp)
        f['lqn'] = np.log10(np.maximum(g.qc.values - sv, 1e-3) / svp)
        for w in WINS:
            k = int(round(w / 0.01)) | 1
            r = lq.rolling(k, center=True, min_periods=1)
            f[f'mean_{w}'] = r.mean().values - lq.values
            f[f'std_{w}'] = r.std().fillna(0).values
            f[f'min_{w}'] = r.min().values - lq.values
            f[f'max_{w}'] = r.max().values - lq.values
        f['grad'] = np.gradient(lq.rolling(11, center=True, min_periods=1).mean().values, g.z.values)
        f['lat'] = m.lat; f['lon'] = m.lon
        f['site'] = m.site; f['region'] = m.region
        f['y'] = np.log10(np.maximum(g.fs.values, 1e-4))
        f['Ic'] = g.Ic.values; f['qc'] = g.qc.values; f['fs'] = g.fs.values
        f['svp_true'] = g.svp.values; f['sv_true'] = g.sv.values
        rows.append(f.iloc[::step])
    return pd.concat(rows, ignore_index=True)


if __name__ == '__main__':
    F = build()
    F.to_pickle('features.pkl')
    print(F.shape); print(F.y.describe())
