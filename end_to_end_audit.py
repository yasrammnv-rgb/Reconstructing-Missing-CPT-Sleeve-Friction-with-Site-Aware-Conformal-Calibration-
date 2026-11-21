"""Independent methodological audit of site-grouped out-of-fold predictions.
Reconstructs density and vertical stresses using predicted fs, not measured fs;
checks predrill infill sensitivity; computes site-balanced diagnostics.
The reference remains an Ic-derived pseudo-label, NOT laboratory soil truth.
"""
from pathlib import Path
import numpy as np,pandas as pd
from sbt import sbt_profile
from ml import zone
ROOT=Path(__file__).parent
p=ROOT/'cv_predictions.pkl'
if not p.exists():raise SystemExit('Main ML rerun must finish first; no cv_predictions.pkl')
cv=pd.read_pickle(p)
meta=pd.read_csv(ROOT/'meta.csv').set_index('ID')
new={name:np.full(len(cv),np.nan) for name in ['rf_endIc','const_endIc','ref_resampledIc']}
for ID,inds in cv.groupby('ID',sort=False).groups.items():
 idx=np.asarray(list(inds));g=cv.loc[idx].sort_values('z');ii=g.index.values
 z=g.z.to_numpy();qc=g.qc.to_numpy();gwt=float(meta.loc[ID,'gwt'])
 for name,fs in [('rf_endIc',10**g.p_rf.to_numpy()),('const_endIc',10**g.p_const_Rf.to_numpy()),('ref_resampledIc',g.fs.to_numpy())]:
  if len(z)<2:continue
  new[name][ii]=sbt_profile(z,qc,fs,gwt)['Ic']
for key,x in new.items():cv[key]=x
print('computed all profile-wise normalized SBT',len(cv),flush=True)
z_ref=zone(cv.Ic.to_numpy())
measures={
 'RF_fixed_stress':np.mean(zone(cv.Ic_rf.values)==z_ref),
 'RF_recomputed_stress':np.mean(zone(cv.rf_endIc.values)==z_ref),
 'constantRf_fixed_stress':np.mean(zone(cv.Ic_const_Rf.values)==z_ref),
 'constantRf_recomputed_stress':np.mean(zone(cv.const_endIc.values)==z_ref),
 'reference_resampled_stress_agreement':np.mean(zone(cv.ref_resampledIc.values)==z_ref),
 'RF_all_rmse':np.sqrt(np.mean((cv.p_rf-cv.y)**2)),
 'RF_no_predrill_rmse':np.nan,
 'RF_predrill_infill_rmse':np.nan,
 'RF_no_predrill_SBTacc':np.nan,
 'RF_predrill_infill_SBTacc':np.nan,
 'RF_site_balanced_accuracy_fixed_stress':np.nan,
 'RF_site_balanced_accuracy_recomputed_stress':np.nan,
 'point_CQR_80_20_cv_coverage':np.mean((cv.y>=cv.lo)&(cv.y<=cv.hi)),
 'site_median_CQR_coverage':np.nan,
 'site_fraction_CQR_below_80percent_coverage':np.nan,
}
cv['predrill']=cv.ID.map(meta.predrill);notpre=cv.z>=cv.predrill
for flag,mask in [('no_predrill',notpre),('predrill_infill',~notpre)]:
 measures['RF_'+flag+'_rmse']=np.sqrt(np.mean((cv.loc[mask,'p_rf']-cv.loc[mask,'y'])**2)) if mask.any() else np.nan
 measures['RF_'+flag+'_SBTacc']=np.mean(zone(cv.loc[mask,'Ic_rf'].to_numpy())==z_ref[mask.to_numpy()]) if mask.any() else np.nan
cv['ok_fixed']=zone(cv.Ic_rf.values)==z_ref
cv['ok_end']=zone(cv.rf_endIc.values)==z_ref
cv['cov']=((cv.y>=cv.lo)&(cv.y<=cv.hi)).astype(float)
grp=cv.groupby('site')[['ok_fixed','ok_end','cov']].mean()
measures['RF_site_balanced_accuracy_fixed_stress']=grp.ok_fixed.mean();measures['RF_site_balanced_accuracy_recomputed_stress']=grp.ok_end.mean()
measures['site_median_CQR_coverage']=grp['cov'].median()
measures['site_fraction_CQR_below_80percent_coverage']=(grp['cov']<.8).mean()
measures['count_predrill_infilled_retained']=int((~notpre).sum())
measures['count_points_retained']=len(cv)
out=pd.DataFrame({'measure':list(measures),'value':list(measures.values())});out.to_csv(ROOT/'end_to_end_audit.csv',index=False)
# Small enough to archive for reproduction and external recalculations
cv[['ID','site','z','Ic','Ic_rf','rf_endIc','Ic_const_Rf','const_endIc','ref_resampledIc','predrill','p_rf','p_const_Rf','y','ok_fixed','ok_end','cov']].to_csv(ROOT/'end_to_end_point_predictions.csv.gz',index=False,compression='gzip')
site=grp.reset_index();site.to_csv(ROOT/'end_to_end_site_metrics.csv',index=False)
print(out.to_string(index=False),flush=True)
