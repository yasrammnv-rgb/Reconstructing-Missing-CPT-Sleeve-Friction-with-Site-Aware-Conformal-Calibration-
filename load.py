import scipy.io as sio, numpy as np, pandas as pd
from pathlib import Path
import os

MAT = os.environ.get('CPT_DATA_PATH', str(Path(__file__).resolve().parent / 'dataset' / 'NorthAmerica_Dataset_V1.mat'))


def load():
    m = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)['NorthAmerica']
    meta, prof = [], []
    for s in m:
        d = np.atleast_1d(np.asarray(s.depth_m, float))
        qc = np.atleast_1d(np.asarray(s.qc_mpa, float))
        fs = np.atleast_1d(np.asarray(s.fs_mpa, float))
        meta.append(dict(ID=int(s.ID), lat=float(s.latitude), lon=float(s.longitude),
                         gwt=float(s.gwt_m) if np.size(s.gwt_m) else np.nan,
                         dz=float(s.depth_interval_m) if np.size(s.depth_interval_m) else np.nan,
                         predrill=float(s.predrill_m) if np.size(s.predrill_m) else np.nan,
                         n=len(d), zmax=np.nanmax(d) if len(d) else np.nan))
        prof.append(pd.DataFrame(dict(ID=int(s.ID), z=d, qc=qc, fs=fs)))
    return pd.DataFrame(meta), pd.concat(prof, ignore_index=True)


if __name__ == '__main__':
    meta, P = load()
    print(meta.describe().T)
    print(P.describe().T)
    print('gwt nan frac', meta.gwt.isna().mean(), 'unique dz', meta.dz.value_counts().head())
    print('neg/zero qc', (P.qc <= 0).mean(), 'neg fs', (P.fs < 0).mean())
    meta.to_csv('meta.csv', index=False); P.to_parquet('profiles.parquet') if False else P.to_pickle('profiles.pkl')
