from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
PROJECT=Path(__file__).resolve().parents[2]; FIG=PROJECT/'paper'/'figures'; FIG.mkdir(exist_ok=True)
COL={'random':'#475569','fastest':'#D97706','loss':'#7C3AED','flcontrol':'#0F766E'}
LAB={'random':'Random','fastest':'Fastest','loss':'Loss-only','flcontrol':'FL-Control'}
s=pd.read_csv(PROJECT/'results'/'reference_run'/'seed_summary.csv'); cal=pd.read_csv(PROJECT/'results'/'reference_run'/'calibration.csv')
# Pareto
agg=s.groupby('policy').agg(f1=('final_f1','mean'),slo=('slo_rate','mean'),work=('total_parallel_epochs','mean')).reset_index()
fig,ax=plt.subplots(figsize=(6.6,4.0))
for _,r in agg.iterrows():
    m=r.policy; ax.scatter(r.slo*100,r.f1,s=80+1.5*r.work,color=COL[m],alpha=.9,edgecolor='white',linewidth=.8); ax.annotate(LAB[m],(r.slo*100,r.f1),xytext=(6,5),textcoords='offset points',fontsize=9)
ax.set_xlabel('Compute-SLO satisfaction (%)'); ax.set_ylabel('Final macro-F1'); ax.set_xlim(65,102); ax.set_ylim(0,.31); ax.grid(alpha=.2)
fig.tight_layout();
for e in ['pdf','png']: fig.savefig(FIG/f'slo_quality_pareto.{e}',dpi=240,bbox_inches='tight')
plt.close(fig)
# Calibration workload (use one copy per seed; calibration repeated across policies)
c=cal[cal.policy=='random']; g=c.groupby('user').epochs.agg(['mean','std']).sort_values('mean')
fig,ax=plt.subplots(figsize=(6.7,3.7)); xs=np.arange(len(g)); ci=1.96*g['std'].to_numpy()/np.sqrt(10)
ax.bar(xs,g['mean'],color='#0F766E',alpha=.9); ax.errorbar(xs,g['mean'],yerr=ci,fmt='none',ecolor='#111827',capsize=3,lw=1)
ax.set_xticks(xs); ax.set_xticklabels([f'User {u}' for u in g.index]); ax.set_ylabel('Epochs to 20% local-loss reduction'); ax.set_xlabel('Natural WISDM client'); ax.grid(axis='y',alpha=.2)
fig.tight_layout();
for e in ['pdf','png']: fig.savefig(FIG/f'calibrated_client_workload.{e}',dpi=240,bbox_inches='tight')
plt.close(fig)
# SLO and F1 paired bars
order=['fastest','flcontrol','random','loss']; fig,ax1=plt.subplots(figsize=(7.0,3.9)); x=np.arange(len(order)); width=.36
f1=[s[s.policy==m].final_f1.mean() for m in order]; slo=[s[s.policy==m].slo_rate.mean()*100 for m in order]
ax1.bar(x-width/2,f1,width,color=[COL[m] for m in order],alpha=.92,label='Macro-F1'); ax1.set_ylabel('Final macro-F1'); ax1.set_ylim(0,.31); ax1.set_xticks(x); ax1.set_xticklabels([LAB[m] for m in order]); ax1.grid(axis='y',alpha=.15)
ax2=ax1.twinx(); ax2.bar(x+width/2,slo,width,facecolor='none',edgecolor=[COL[m] for m in order],linewidth=2,label='SLO %'); ax2.set_ylabel('SLO satisfaction (%)'); ax2.set_ylim(0,105)
fig.tight_layout();
for e in ['pdf','png']: fig.savefig(FIG/f'quality_and_slo.{e}',dpi=240,bbox_inches='tight')
plt.close(fig)
