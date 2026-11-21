"""Site-aware conformal quantile regression, replicating the parent 75/25 site split.
Guarantee requires exchangeability of whole site groups, not presumed in spatial transfer.
"""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source_code'))
from ml import hgb,ALPHA
FEATURE_FILE = HERE/'features.pkl' if (HERE/'features.pkl').exists() else HERE/'source_code'/'features.pkl'
F=pd.read_pickle(FEATURE_FILE);F=F[F.fs>5e-4].reset_index(drop=True)
feat=[x for x in F if x not in ('ID','site','region','y','Ic','qc','fs','svp_true','sv_true')]
y=F.y.to_numpy();group=F.site.to_numpy()
methods=['raw','point_cqr','site_90_90','site_simultaneous']
res={key:[np.full(len(F),np.nan),np.full(len(F),np.nan)] for key in methods}
folds=np.zeros(len(F),dtype=int);qrows=[]
def conformal_q(s,alpha=.1):
  s=np.sort(np.asarray(s));k=int(np.ceil((len(s)+1)*(1-alpha)))
  return float(s[k-1]) if k<=len(s) else float('inf')
def pct_higher(s,p):
  s=np.sort(np.asarray(s));return float(s[int(np.ceil(len(s)*p))-1])
for k,(tr,te) in enumerate(GroupKFold(5).split(F,y,group)):
  t0=time.time()
  ptr,cal=next(GroupShuffleSplit(1,test_size=.25,random_state=k).split(tr,y[tr],group[tr]));ptr,cal=tr[ptr],tr[cal]
  qlo=hgb(loss='quantile',quantile=ALPHA/2).fit(F.iloc[ptr][feat],y[ptr]);qhi=hgb(loss='quantile',quantile=1-ALPHA/2).fit(F.iloc[ptr][feat],y[ptr])
  lc,hc=qlo.predict(F.iloc[cal][feat]),qhi.predict(F.iloc[cal][feat]);lt,ht=qlo.predict(F.iloc[te][feat]),qhi.predict(F.iloc[te][feat])
  s=np.maximum(lc-y[cal],y[cal]-hc)
  unique_site=np.unique(group[cal]);s90=[];smax=[]
  for site in unique_site:
    vals=s[group[cal]==site];s90.append(pct_higher(vals,.9));smax.append(float(vals.max()))
  qs={'raw':0.,'point_cqr':conformal_q(s),'site_90_90':conformal_q(s90),'site_simultaneous':conformal_q(smax)}
  for name,q in qs.items():res[name][0][te]=lt-q;res[name][1][te]=ht+q
  folds[te]=k;qrows.append({'fold':k,'train_sites':len(np.unique(group[ptr])),'calibration_sites':len(unique_site),'test_sites':len(np.unique(group[te])),'calibration_points':len(cal),**{'threshold_'+a:b for a,b in qs.items()}})
  print(f'fold={k} seconds={time.time()-t0:.1f} thresholds={qs}',flush=True)
out=F[['ID','site','region','z','y','Ic']].copy();out['fold']=folds
for name,(a,b) in res.items():out['lo_'+name],out['hi_'+name]=a,b
out.to_pickle(HERE/'site_aware_CQR_predictions.pkl')
rows=[];sites=[];regions=[]
for name in methods:
  hit=((out.y>=out['lo_'+name])&(out.y<=out['hi_'+name]));width=out['hi_'+name]-out['lo_'+name]
  tmp=out[['site','region']].copy();tmp['hit']=hit;tmp['width']=width
  S=tmp.groupby('site').agg(coverage=('hit','mean'),width=('width','mean'),n_points=('hit','size'),region=('region','first'))
  rows.append({'method':name,'point_coverage':hit.mean(),'site_mean_coverage':S.coverage.mean(),'site_median_coverage':S.coverage.median(),'frac_sites_ge_90':(S.coverage>=.9-1e-12).mean(),'frac_sites_all_covered':(S.coverage>=1-1e-12).mean(),'mean_width_log10':width.mean(),'n_sites':len(S),'n_points':len(out)})
  for i,r in S.reset_index().iterrows():sites.append(dict(method=name,**r.to_dict()))
  for region,rr in tmp.groupby('region'):
    sub=S[S.region==region];regions.append(dict(method=name,region=region,n_sites=len(sub),n_points=len(rr),point_coverage=rr.hit.mean(),site_fraction_ge_90=(sub.coverage>=.9-1e-12).mean(),full_site_coverage=(sub.coverage>=1-1e-12).mean(),mean_width_log10=rr.width.mean()))
pd.DataFrame(qrows).to_csv(HERE/'site_aware_CQR_calibration.csv',index=False)
pd.DataFrame(rows).to_csv(HERE/'site_aware_CQR_metrics.csv',index=False)
pd.DataFrame(sites).to_csv(HERE/'site_aware_CQR_by_site.csv',index=False)
pd.DataFrame(regions).to_csv(HERE/'site_aware_CQR_by_region.csv',index=False)
import scipy,sklearn
(HERE/'site_aware_CQR_software.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,scikit_learn=sklearn.__version__,method='GroupKFold(5), GroupShuffleSplit(test_size=0.25, random_state=outer_fold); exact k-th order statistic across calibration groups',alpha=ALPHA),indent=2))
print(pd.DataFrame(rows).round(4).to_string(index=False),flush=True)
