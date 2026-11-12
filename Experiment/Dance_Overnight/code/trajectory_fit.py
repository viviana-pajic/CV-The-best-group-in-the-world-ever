"""Offline robust trajectory fit. No detector/identity changes or hidden-point labels."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from filter_lab import path_outliers

def fit(z,confidence,cfg):
    zz=path_outliers(z,confidence,cfg.get('path_speed',.3)) if cfg.get('path_gate') else z.copy()
    a=np.full_like(z,np.nan);state=np.zeros(z.shape[:2],int);age=np.full(z.shape[:2],999,int)
    for j in range(z.shape[1]):
        kk=np.flatnonzero(np.isfinite(zz[:,j,0]));segments=np.split(kk,np.flatnonzero(np.diff(kk)>cfg.get('gap',25))+1)
        for ix in segments:
            if not len(ix):continue
            first,last=ix[0],ix[-1];n=last-first+1;t=ix-first;mask=np.zeros(n,bool);mask[t]=True;target=np.zeros((n,2));target[t]=zz[ix,j];weights=np.zeros(n);weights[t]=np.clip(confidence[ix,j],.1,1.)
            if n<=2:out=target
            else:
                d=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc');regularizer=cfg['strength']*(d.T@d);w=weights.copy()
                for it in range(8):
                    mat=regularizer+sparse.diags(w+1e-8,format='csc');out=spsolve(mat,w[:,None]*target);err=np.linalg.norm(out-target,axis=1);w=weights*np.minimum(1.,cfg.get('huber',1.)/np.maximum(err,1e-6))
            for q in range(n):
                nearest=int(np.min(np.abs(t-q)));p=out[q]
                if nearest>cfg.get('gap',25) or not(0<=p[0]<=100 and 0<=p[1]<=62.5):continue
                residual=np.linalg.norm(p-target[q]) if mask[q] else np.inf
                a[first+q,j]=p;state[first+q,j]=1 if mask[q] and residual<=cfg.get('measured_residual',3.) else 2;age[first+q,j]=nearest
    return a,state,age

def configs():
    for gate in [False,True]:
        for strength in [2,8,32,128,512]:
            for huber in [.5,1.5]:
                yield {'name':f'robust_l{strength}_h{huber}_path{int(gate)}','kind':'robust_trajectory','strength':strength,'huber':huber,'path_gate':gate,'path_speed':.3,'gap':25,'measured_residual':3.}
