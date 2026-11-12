"""A trained non-image baseline: can box shape alone explain the improvement?"""
import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from candidate_evaluate import ROOT,OUT,read,write,oracle_scores

def features(box):
    x1,y1,x2,y2=np.asarray(box,float);w=max(x2-x1,10);h=max(y2-y1,10);cx=(x1+x2)/2560;cy=(y1+y2)/1440;ratio=w/h;lh=np.log(h)
    return np.array([cx,cy,w/1280,h/720,ratio,lh,cx*cy,cy*cy,ratio*ratio,lh*lh,cy*ratio,lh*ratio])

def fit(rows,ridge):
    x=np.array([features(o['box']) for o in rows]);mu=x.mean(0);sd=x.std(0).clip(.001);X=np.c_[np.ones(len(x)),(x-mu)/sd];visible=np.array([o['floor'] is not None for o in rows]);Y=[]
    for o in rows:
        if o['floor'] is None:continue
        a,b,c,d=o['box'];Y.append([(o['floor'][0]-(a+c)/2)/max(c-a,10),(o['floor'][1]-d)/max(d-b,10)])
    A=X[visible];reg=np.eye(X.shape[1])*ridge;reg[0,0]=0;beta=np.linalg.solve(A.T@A+reg,A.T@np.array(Y));v=visible.astype(float)
    def loss(w):
        logits=X@w;p=expit(logits);return np.logaddexp(0,logits).sum()-v@logits+.5*ridge*(w[1:]@w[1:]),X.T@(p-v)+np.r_[0,ridge*w[1:]]
    result=minimize(loss,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500});assert result.success,result.message
    return {'mean':mu.tolist(),'scale':sd.tolist(),'point_coefficients':beta.tolist(),'visibility_coefficients':result.x.tolist(),'ridge':ridge}

def predict(model,box):
    X=np.r_[1,(features(box)-model['mean'])/model['scale']];q=X@np.asarray(model['point_coefficients']);a,b,c,d=box;point=[(a+c)/2+q[0]*(c-a),d+q[1]*(d-b)]
    return {'point':point,'peak_point':point,'reportability':float(expit(X@model['visibility_coefficients'])),'augmentation_rms_pixels':0.}

def main():
    root=OUT/'geometry';root.mkdir(exist_ok=True);manifest=read(OUT/'data/development_expanded.json');train=[o for d in manifest.values() for k in d['train'] for o in d['labels'][str(k)] if o['box'] is not None and o['contact']!='uncertain'];val=read(OUT/'data/development_v2.json')
    for d in val.values():
        for k in d['val']:d['labels'][str(k)]=[o for o in d['labels'][str(k)] if o['contact']!='uncertain']
    results=[]
    for penalty in [.1,1.,10.,100.]:
        model=fit(train,penalty);pred={f'{clip}:{k}':{str(o['id']):predict(model,o['box']) for o in d['labels'][str(k)] if o['box'] is not None} for clip,d in val.items() for k in d['val']};score=oracle_scores(pred,val);results.append({'model':model,'validation':score});write(root/f'ridge_{penalty}_validation.json',pred)
    chosen=max(results,key=lambda r:r['validation']['selected']['joint_f1']);write(root/'results.json',{'scope':'Development-only baseline trained on same913 clean crop labels, using bounding-box geometry without image pixels or time/identity features.','variants':results,'chosen':chosen});print('Geometry baseline',chosen['validation']['selected'],flush=True)
    base=read(OUT/'tracking/flow_b1.0_gap1_a0.5_tracks.json');out=OUT/'pipelines/geometry_flow';out.mkdir(parents=True,exist_ok=True);write(out/'predictions.json',[{str(o['id']):predict(chosen['model'],o['box']) for o in row if o['status']=='observed'} for row in base['frames']]);write(out/'base_tracks.json',base);write(out/'settings.json',chosen)

if __name__=='__main__':main()
