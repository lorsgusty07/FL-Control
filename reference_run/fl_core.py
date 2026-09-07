from __future__ import annotations
import copy, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1)  # deterministic, low-overhead execution for the small benchmark model
from sklearn.metrics import f1_score, accuracy_score

FEATURES=['x_absdev','y_absdev','z_absdev','x_std','y_std','z_std','resultant']
LABELS=['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']
LABEL_TO_ID={x:i for i,x in enumerate(LABELS)}

class SensorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(7,16),nn.ReLU(),nn.Linear(16,6))
    def forward(self,x): return self.net(x)

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def load_clients(csv_path:str|Path, warmup:int=4):
    df=pd.read_csv(csv_path).sort_values(['user','unique_id']).reset_index(drop=True)
    # fit scaler ONLY on warm-up windows; no future-window statistics are used
    warm=df.groupby('user',sort=True).head(warmup)
    mu=warm[FEATURES].mean().to_numpy(np.float32)
    sd=warm[FEATURES].std(ddof=0).replace(0,1).to_numpy(np.float32)
    out={}
    for user,g in df.groupby('user',sort=True):
        X=((g[FEATURES].to_numpy(np.float32)-mu)/sd).astype(np.float32)
        y=np.array([LABEL_TO_ID[v] for v in g.label],dtype=np.int64)
        out[int(user)]={'X':X,'y':y,'labels':g.label.tolist(),'ids':g.unique_id.astype(int).tolist()}
    return out,mu,sd

def model_bytes(model:nn.Module):
    return sum(p.numel()*p.element_size() for p in model.state_dict().values())

def local_train(global_model, X, y, seed, epochs=5, lr=0.04):
    model=copy.deepcopy(global_model)
    gen=torch.Generator().manual_seed(seed)
    opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.0)
    xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.long)
    n=len(yt)
    for ep in range(epochs):
        order=torch.randperm(n,generator=gen)
        for idx in order.split(max(1,min(8,n))):
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(xt[idx]),yt[idx]); loss.backward(); opt.step()
    return {k:v.detach().clone() for k,v in model.state_dict().items()}

def fedavg(states,weights):
    total=float(sum(weights)); out={}
    for k in states[0]:
        out[k]=sum(s[k]*(w/total) for s,w in zip(states,weights))
    return out

def evaluate(model,X,y):
    model.eval()
    with torch.no_grad():
        logits=model(torch.tensor(X,dtype=torch.float32)); pred=logits.argmax(1).cpu().numpy()
    return pred

def local_loss(model,X,y):
    model.eval()
    with torch.no_grad():
        return float(F.cross_entropy(model(torch.tensor(X,dtype=torch.float32)),torch.tensor(y,dtype=torch.long)).item())

def grad_norm(model,X,y):
    m=copy.deepcopy(model); m.zero_grad(set_to_none=True)
    loss=F.cross_entropy(m(torch.tensor(X,dtype=torch.float32)),torch.tensor(y,dtype=torch.long)); loss.backward()
    return float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in m.parameters() if p.grad is not None)).item())

def entropy(labels, nclasses=6):
    c=np.bincount(np.asarray(labels,dtype=int),minlength=nclasses).astype(float); p=c/c.sum()
    p=p[p>0]
    return float(-(p*np.log(p)).sum()/np.log(nclasses)) if len(p) else 0.0

def js_divergence(a,b,nclasses=6):
    pa=np.bincount(np.asarray(a,dtype=int),minlength=nclasses).astype(float)+1e-12
    pb=np.bincount(np.asarray(b,dtype=int),minlength=nclasses).astype(float)+1e-12
    pa/=pa.sum(); pb/=pb.sum(); m=.5*(pa+pb)
    kl=lambda p,q: float(np.sum(p*np.log(p/q)))
    return .5*kl(pa,m)+.5*kl(pb,m)

def norm(v):
    a=np.asarray(v,float); d=a.max()-a.min()
    return np.zeros_like(a) if d<1e-12 else (a-a.min())/d

import time

def adaptive_local_train(global_model, X, y, seed, rel_reduction=0.20, max_epochs=25, lr=0.035):
    model=copy.deepcopy(global_model)
    gen=torch.Generator().manual_seed(seed)
    opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.0)
    xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.long)
    start=local_loss(model,X,y); target=start*(1-rel_reduction)
    t0=time.perf_counter(); epochs=0
    for ep in range(1,max_epochs+1):
        order=torch.randperm(len(yt),generator=gen)
        for idx in order.split(max(1,min(8,len(yt)))):
            opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(xt[idx]),yt[idx]); loss.backward(); opt.step()
        epochs=ep
        if local_loss(model,X,y) <= target: break
    elapsed=time.perf_counter()-t0
    return {k:v.detach().clone() for k,v in model.state_dict().items()},epochs,elapsed,start,local_loss(model,X,y)
