import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import figs
T=pd.read_csv('cqr_table.csv').set_index('method')
regs=['Pacific','Central/Interior','South Atlantic','Gulf Coast','North Atlantic','Mountain/Plains','Alaska/Canada']
lab={'Alaska/Canada':'Other\n(AK/HI/CA)'}
C=['#2a78d6','#eb6834','#1baf7a']; nm={'raw':'Raw quantile GBDT','cqr':'Split CQR','mondrian':'Mondrian CQR (by region)'}
fig,ax=plt.subplots(figsize=(7.2,3)); w=0.26; x=np.arange(len(regs)+1)
for j,m in enumerate(['raw','cqr','mondrian']):
    v=[T.loc[m,'coverage']]+[T.loc[m,'cov_'+r] for r in regs]
    ax.bar(x+(j-1)*w,v,w*0.92,color=C[j],label=nm[m])
ax.axhline(0.9,color='#52514e',lw=1,ls='--'); ax.text(len(regs)+0.45,0.905,'nominal 0.90',fontsize=7,color='#52514e',ha='right',va='bottom')
ax.set_xticks(x); ax.set_xticklabels(['All']+[lab.get(r,r.replace('/','/\n')) for r in regs],fontsize=7.5)
ax.set_ylim(0.4,1.0); ax.set_ylabel('Empirical coverage'); ax.grid(axis='x',visible=False)
ax.legend(fontsize=7.5,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.15))
fig.tight_layout(); fig.savefig('fig6_coverage.png')
