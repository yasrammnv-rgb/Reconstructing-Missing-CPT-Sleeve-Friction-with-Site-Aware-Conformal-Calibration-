"""Region-held-out RF stress test, independent of leave-site-out GroupKFold.
Seven region folds, exact RF hyperparameters and subsample size of original script.
Only supervised fs prediction features; provides end-to-end and fixed-reference-stress metrics.
"""
import pandas as pd,numpy as np, time
from sklearn.ensemble import RandomForestRegressor
from ml import zone, ic_from
from sbt import sbt_profile
F=pd.read_pickle('features.pkl');F=F[F.fs>5e-4].reset_index(drop=True)
features=[c for c in F.columns if c not in ('ID','site','region','y','Ic','qc','fs','svp_true','sv_true')]
res=[];out=np.empty(len(F)); base=np.empty(len(F));
for idx,reg in enumerate(F.region.unique()):
 t=time.time();tr=np.flatnonzero(F.region.values!=reg);te=np.flatnonzero(F.region.values==reg)
 sub=np.random.default_rng(idx+42).choice(tr,size=min(120000,len(tr)),replace=False)
 rf=RandomForestRegressor(n_estimators=150,min_samples_leaf=10,max_features=.4,n_jobs=2,random_state=0)
 rf.fit(F.iloc[sub][features], F.y.values[sub]);p=rf.predict(F.iloc[te][features])
 b=F.lqc.values[te]+np.median(F.y.values[tr]-F.lqc.values[tr]);out[te]=p;base[te]=b
 Z=zone(F.Ic.values[te]);icp=ic_from(F.qc.values[te],10**p,F.sv_true.values[te],F.svp_true.values[te])
 row={'region':reg,'n_sites':F.iloc[te].site.nunique(),'n_points':len(te),'RMSE_rf':np.sqrt(np.mean((p-F.y.values[te])**2)),'RMSE_constRf':np.sqrt(np.mean((b-F.y.values[te])**2)),'SBTacc_RF_fixed_stress':np.mean(zone(icp)==Z),'seconds':round(time.time()-t,1)}
 res.append(row);print(row,flush=True)
F[['ID','site','region','z','qc','fs','y','Ic','sv_true','svp_true']].assign(pred_rf_regionout=out,pred_constRf_regionout=base).to_pickle('regionout_rf_predictions.pkl')
D=pd.DataFrame(res).sort_values('region');D.to_csv('regionout_rf_results.csv',index=False)
print('pooled all',np.sqrt(np.mean((out-F.y.values)**2)),np.sqrt(np.mean((base-F.y.values)**2)))
