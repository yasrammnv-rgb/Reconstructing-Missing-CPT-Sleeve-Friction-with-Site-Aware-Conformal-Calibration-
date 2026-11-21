from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
M=pd.read_csv(D/'site_aware_CQR_metrics.csv')
labels=['Raw quantile','Point CQR','Site 90/90','Site max']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'savefig.dpi':400})
fig,axs=plt.subplots(1,2,figsize=(11.4,4.2),gridspec_kw={'width_ratios':[1.5,1]})
x=np.arange(4);w=.34
bar1=axs[0].bar(x-w/2,100*M.frac_sites_ge_90,width=w,label='≥90% of depth points')
bar2=axs[0].bar(x+w/2,100*M.frac_sites_all_covered,width=w,label='Entire grouped site')
axs[0].axhline(90,ls='--',lw=1,color='0.3',label='90% of sites')
axs[0].set_xticks(x,labels);axs[0].set_ylim(0,110);axs[0].set_ylabel('Fraction of held-out site groups (%)');axs[0].set_title('(a) Attainment of within-site coverage targets');axs[0].grid(axis='y',alpha=.22)
axs[0].legend(loc='upper left',fontsize=8)
for b in list(bar1)+list(bar2):
  v=b.get_height();axs[0].annotate(f'{v:.1f}',(b.get_x()+b.get_width()/2,v),xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=8)
axs[1].bar(x,M.mean_width_log10)
axs[1].set_xticks(x,labels);axs[1].set_ylabel('Mean interval width (log10 fs)');axs[1].set_title('(b) Cost of profile-aware calibration');axs[1].grid(axis='y',alpha=.22)
for i,v in enumerate(M.mean_width_log10):axs[1].text(i,v+.05,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
axs[1].set_ylim(0,3.7)
for ax in axs:
 for tick in ax.get_xticklabels():tick.set_rotation(16);tick.set_ha('right')
fig.tight_layout(pad=1.4)
fig.savefig(D/'figure12_site_aware_CQR.png',dpi=400,bbox_inches='tight')
fig.savefig(D/'figure12_site_aware_CQR.pdf',bbox_inches='tight')
plt.close(fig)
