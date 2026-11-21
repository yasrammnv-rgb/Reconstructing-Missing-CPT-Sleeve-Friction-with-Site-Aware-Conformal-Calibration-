"""Independent leave-one-region-out *linear* diagnostic. NOT an RF external-validation claim.
Ridge with no coordinates; train-only baseline; fixed measured-fs-derived stress evaluation
reported distinctly; evaluates regional transfer sensitivity in publicly provided V1 archive.
"""
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from ml import zone,ic_from
F=pd.read_pickle('features.pkl');F=F[F.fs>5e-4].reset_index(drop=True)
feat=[c for c in F if c not in ('ID','site','region','y','Ic','qc','fs','svp_true','sv_true','lat','lon')]
rows=[]; pred=np.empty(len(F));baseline=np.empty(len(F))
for reg in F.region.unique():
 tr=np.flatnonzero(F.region.values!=reg);te=np.flatnonzero(F.region.values==reg)
 Xtr=F.iloc[tr][feat].to_numpy(dtype='float64');Xte=F.iloc[te][feat].to_numpy(dtype='float64')
 scal=StandardScaler().fit(Xtr);model=Ridge(alpha=10).fit(scal.transform(Xtr),F.y.values[tr])
 p=model.predict(scal.transform(Xte));b=F.lqc.values[te]+np.median(F.y.values[tr]-F.lqc.values[tr]);pred[te]=p;baseline[te]=b
 icp=ic_from(F.qc.values[te],10**p,F.sv_true.values[te],F.svp_true.values[te]);icb=ic_from(F.qc.values[te],10**b,F.sv_true.values[te],F.svp_true.values[te])
 zt=zone(F.Ic.values[te]);
 rows.append(dict(region=reg,n_points=len(te),n_sites=F.iloc[te].site.nunique(),RMSE_ridge=np.sqrt(np.mean((p-F.y.values[te])**2)),RMSE_const_Rf=np.sqrt(np.mean((b-F.y.values[te])**2)),SBT_ridge_fixed_stress=np.mean(zone(icp)==zt),SBT_const_Rf_fixed_stress=np.mean(zone(icb)==zt)))
 print(rows[-1],flush=True)
out=pd.DataFrame(rows).sort_values('region');out.to_csv('regional_ridge_stress_test.csv',index=False)
F[['ID','z','region','site','y','Ic','qc','fs','sv_true','svp_true']].assign(pred_regionout_ridge=pred,pred_regionout_const_Rf=baseline).to_pickle('regional_ridge_predictions.pkl')
print('pooled',np.sqrt(np.mean((pred-F.y.values)**2)),np.sqrt(np.mean((baseline-F.y.values)**2)))
