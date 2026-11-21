"""Deterministic site manifests for all evaluation/calibration folds."""
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
HERE=Path(__file__).resolve().parent
FEATURE_FILE = HERE/'features.pkl' if (HERE/'features.pkl').exists() else HERE/'source_code'/'features.pkl'
F=pd.read_pickle(FEATURE_FILE);F=F[F.fs>5e-4].reset_index(drop=True)
g=F.site.to_numpy();y=F.y.to_numpy()
rows=[]
for k,(tr,te) in enumerate(GroupKFold(5).split(F,y,g)):
 ptr,cal=next(GroupShuffleSplit(1,test_size=.25,random_state=k).split(tr,y[tr],g[tr]));ptr,cal=tr[ptr],tr[cal]
 for label,idx in [('proper_training',ptr),('calibration',cal),('test',te)]:
  subset=F.iloc[idx]
  stat=subset.groupby('site').agg(n_depth_points=('y','size'),n_soundings=('ID','nunique'),region=('region','first'),min_depth=('z','min'),max_depth=('z','max'))
  for site,s in stat.iterrows():rows.append(dict(outer_fold=k,site_id=int(site),role=label,**s.to_dict()))
M=pd.DataFrame(rows)
M.to_csv(HERE/'site_split_manifest.csv',index=False)
assert len(M)==4000 and len(M[M.role=='test'])==800
assert (M.groupby(['outer_fold','site_id']).size()==1).all()
assert M.query('role=="test"').site_id.nunique()==800
print(M.groupby(['outer_fold','role']).agg(sites=('site_id','size'),points=('n_depth_points','sum')).to_string())
