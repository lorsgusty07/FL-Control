from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import wilcoxon
from fl_core import *

PROJECT=Path(__file__).resolve().parents[2]; DATA=PROJECT/'data'/'reference_run'/'wisdm_real_6users_12windows.csv'; OUT=PROJECT/'results'/'reference_run'; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[17,29,43,59,71,89,107,131,167,197]
POLICIES=['random','fastest','loss','flcontrol']; K=3; TRAIN_N=8; TEST_START=8; ROUNDS=12

def warm_model(clients, seed):
    seed_all(seed); m=SensorMLP(); users=sorted(clients)
    for r in range(12):
        st=[]; wt=[]
        for j,u in enumerate(users):
            c=clients[u]; st.append(local_train(m,c['X'][:4],c['y'][:4],seed*10000+r*100+j,epochs=3,lr=.04)); wt.append(4)
        m.load_state_dict(fedavg(st,wt))
    return m

def calibrate(model,clients,seed):
    rows=[]
    for j,u in enumerate(sorted(clients)):
        c=clients[u]
        _,ep,sec,l0,l1=adaptive_local_train(model,c['X'][:TRAIN_N],c['y'][:TRAIN_N],seed*1000+j,rel_reduction=.20,max_epochs=25,lr=.035)
        rows.append({'user':u,'epochs':ep,'seconds':sec,'loss0':l0,'loss1':l1})
    return pd.DataFrame(rows)

def choose(policy,model,clients,cost_pred,deadline,rng):
    users=sorted(clients); losses={u:local_loss(model,clients[u]['X'][:TRAIN_N],clients[u]['y'][:TRAIN_N]) for u in users}
    if policy=='random': return list(map(int,rng.choice(users,K,replace=False))),losses
    if policy=='fastest': return sorted(users,key=lambda u:(cost_pred[u],u))[:K],losses
    if policy=='loss': return sorted(users,key=lambda u:(losses[u],-u),reverse=True)[:K],losses
    # explicit constrained optimization over the 20 possible 3-client cohorts
    feasible=[]
    for comb in itertools.combinations(users,K):
        c=max(cost_pred[u] for u in comb)
        if c <= deadline+1e-9: feasible.append((sum(losses[u] for u in comb),-c,comb))
    if feasible: return list(max(feasible)[2]),losses
    # if no complete cohort is feasible, minimize predicted violation then maximize learning utility
    cand=[]
    for comb in itertools.combinations(users,K): cand.append((-max(cost_pred[u] for u in comb),sum(losses[u] for u in comb),comb))
    return list(max(cand)[2]),losses

def run(seed,policy):
    clients,_,_=load_clients(DATA,warmup=4); model=warm_model(clients,seed); cal=calibrate(model,clients,seed)
    # data-derived and fixed before comparing policies: 60th percentile of six observed client workloads
    deadline=float(np.quantile(cal.epochs,0.60,method='higher'))
    cost_pred={int(r.user):float(r.epochs) for r in cal.itertuples()}; sec_per_epoch={int(r.user):float(r.seconds/max(r.epochs,1)) for r in cal.itertuples()}
    rng=np.random.default_rng(seed+sum(map(ord,policy))); rows=[]; bytes_model=model_bytes(model)
    testX=np.vstack([clients[u]['X'][TEST_START:] for u in sorted(clients)]); testy=np.concatenate([clients[u]['y'][TEST_START:] for u in sorted(clients)])
    for rr in range(1,ROUNDS+1):
        selected,losses=choose(policy,model,clients,cost_pred,deadline,rng)
        states=[];weights=[]; actual=[]; secs=[]
        for j,u in enumerate(selected):
            c=clients[u]; st,ep,sec,l0,l1=adaptive_local_train(model,c['X'][:TRAIN_N],c['y'][:TRAIN_N],seed*100000+rr*100+j,rel_reduction=.20,max_epochs=25,lr=.035)
            states.append(st); weights.append(TRAIN_N); actual.append(ep); secs.append(sec)
            cost_pred[u]=.65*cost_pred[u]+.35*ep
        model.load_state_dict(fedavg(states,weights))
        pred=evaluate(model,testX,testy); f1=f1_score(testy,pred,labels=list(range(6)),average='macro',zero_division=0); acc=accuracy_score(testy,pred)
        round_work=sum(actual); parallel_epochs=max(actual); parallel_seconds=max(secs)
        rows.append({'seed':seed,'policy':policy,'round':rr,'f1':f1,'accuracy':acc,'deadline_epochs':deadline,
                     'parallel_epochs':parallel_epochs,'work_epochs':round_work,'parallel_seconds_observed':parallel_seconds,
                     'slo_met':int(parallel_epochs<=deadline),'selected':';'.join(map(str,selected)),
                     'selected_mean_loss':np.mean([losses[u] for u in selected]),'bytes_round':K*bytes_model*2})
    df=pd.DataFrame(rows)
    s={'seed':seed,'policy':policy,'final_f1':df.tail(3).f1.mean(),'best_f1':df.f1.max(),'final_accuracy':df.tail(3).accuracy.mean(),
       'slo_rate':df.slo_met.mean(),'total_work_epochs':df.work_epochs.sum(),'total_parallel_epochs':df.parallel_epochs.sum(),
       'total_parallel_seconds_observed':df.parallel_seconds_observed.sum(),'total_bytes':df.bytes_round.sum(),'deadline_epochs':deadline}
    cal['seed']=seed; return df,s,cal

def holm(pvals):
    names=list(pvals); vals=np.array([pvals[n] for n in names]); order=np.argsort(vals); adj=np.empty_like(vals); running=0; m=len(vals)
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*vals[idx]); running=max(running,val); adj[idx]=running
    return {n:float(adj[i]) for i,n in enumerate(names)}

def main():
    rows=[]; sums=[]; cals=[]
    for seed in SEEDS:
        for p in POLICIES:
            r,s,c=run(seed,p); rows.append(r); sums.append(s); c['policy']=p; cals.append(c); print('done',seed,p,s['final_f1'],s['slo_rate'])
    traj=pd.concat(rows,ignore_index=True); summ=pd.DataFrame(sums); cal=pd.concat(cals,ignore_index=True)
    traj.to_csv(OUT/'round_metrics.csv',index=False); summ.to_csv(OUT/'seed_summary.csv',index=False); cal.to_csv(OUT/'calibration.csv',index=False)
    agg=summ.groupby('policy').agg(final_f1_mean=('final_f1','mean'),final_f1_sd=('final_f1','std'),best_f1_mean=('best_f1','mean'),slo_rate_mean=('slo_rate','mean'),work_epochs_mean=('total_work_epochs','mean'),parallel_epochs_mean=('total_parallel_epochs','mean'),observed_seconds_mean=('total_parallel_seconds_observed','mean'),deadline_epochs_mean=('deadline_epochs','mean')).reset_index()
    agg.to_csv(OUT/'aggregate.csv',index=False)
    tests=[]; pv=summ.pivot(index='seed',columns='policy',values='final_f1'); raw={}
    for b in ['random','fastest','loss']:
        stat,p=wilcoxon(pv.flcontrol,pv[b],zero_method='wilcox'); raw[b]=p
    adj=holm(raw)
    for b in raw:
        diff=pv.flcontrol-pv[b]; tests.append({'metric':'final_f1','comparison':f'flcontrol-vs-{b}','p_raw':raw[b],'p_holm':adj[b],'paired_effect_dz':float(diff.mean()/(diff.std(ddof=1)+1e-12)),'mean_diff':float(diff.mean())})
    ps=summ.pivot(index='seed',columns='policy',values='slo_rate'); raw2={}
    for b in ['random','fastest','loss']:
        d=ps.flcontrol-ps[b]
        if np.allclose(d,0): p=1.0
        else: _,p=wilcoxon(ps.flcontrol,ps[b],zero_method='wilcox')
        raw2[b]=p
    adj2=holm(raw2)
    for b in raw2:
        d=ps.flcontrol-ps[b]; tests.append({'metric':'slo_rate','comparison':f'flcontrol-vs-{b}','p_raw':raw2[b],'p_holm':adj2[b],'paired_effect_dz':float(d.mean()/(d.std(ddof=1)+1e-12)),'mean_diff':float(d.mean())})
    pd.DataFrame(tests).to_csv(OUT/'paired_tests.csv',index=False)
    config={'dataset':'WISDM v1.1 transformed real-window trace subset','users':[13,17,20,27,29,33],'train_windows_per_user':8,'test_windows_per_user':4,'seeds':SEEDS,'K':K,'rounds':ROUNDS,'policies':POLICIES,'model':'MLP 7-16-6','local_service':'train until 20% local-loss reduction or 25 epochs','SLO':'parallel-equivalent local epochs <= seed-specific 60th percentile of real calibration costs','no_synthetic_resources':True}
    (OUT/'experiment_config.json').write_text(json.dumps(config,indent=2))
    print('\nCAL mean epochs by user\n',cal.groupby('user').epochs.mean().to_string()); print('\nAGG\n',agg.to_string(index=False)); print('\nTESTS\n',pd.DataFrame(tests).to_string(index=False))
if __name__=='__main__': main()
