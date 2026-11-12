"""Development-only map filters. Never modifies detector boxes or identity association."""
import sys,json,copy,itertools,argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
from compare import stage,match_boxes
from occlusion_pilot import inside
from evaluate_support import read,write

def inputs(base,pred,threshold):
    n=len(base['frames']);ids=sorted({o['id'] for r in base['frames'] for o in r});idx={v:i for i,v in enumerate(ids)}
    z=np.full((n,len(ids),2),np.nan);confidence=np.zeros((n,len(ids)))
    for k,row in enumerate(base['frames']):
        for o in row:
            p=pred[k].get(str(o['id']),pred[k].get(o['id']));j=idx[o['id']]
            if o['status']=='observed' and p and p['reportability']>=threshold:
                v=stage(p['point'])
                if inside(v):z[k,j]=v;confidence[k,j]=p['reportability']
    return z,confidence,ids

def original_arrays(base,ids):
    n=len(base['frames']);a=np.full((n,len(ids),2),np.nan);state=np.zeros((n,len(ids)),int);idx={v:i for i,v in enumerate(ids)}
    for k,row in enumerate(base['frames']):
        for o in row:
            if o['map'] is not None:a[k,idx[o['id']]]=o['map'];state[k,idx[o['id']]]=1 if o['map_status']=='proxy' else 2
    return a,state

def isolated_outliers(z,limit=3.):
    out=z.copy()
    for j in range(z.shape[1]):
        for k in range(2,len(z)-2):
            before=z[k-2:k,j];after=z[k+1:k+3,j]
            if np.isfinite(before).all() and np.isfinite(after).all() and np.isfinite(z[k,j]).all():
                p=np.median(before,axis=0);q=np.median(after,axis=0)
                if np.linalg.norm(p-q)<1.5 and np.linalg.norm(z[k,j]-(p+q)/2)>limit:out[k,j]=np.nan
    return out

def hampel_outliers(z,window=9,limit=3.):
    """Offline short-burst rejection; absent frames are never treated as zero."""
    out=z.copy();half=window//2
    for j in range(z.shape[1]):
        for k in range(len(z)):
            if not np.isfinite(z[k,j]).all():continue
            values=z[max(0,k-half):min(len(z),k+half+1),j];values=values[np.isfinite(values).all(1)]
            if len(values)<4:continue
            center=np.median(values,axis=0);dist=np.linalg.norm(values-center,axis=1);scale=max(.2,1.4826*np.median(dist))
            if np.linalg.norm(z[k,j]-center)>max(limit,3*scale):out[k,j]=np.nan
    return out

def path_outliers(z,confidence,max_speed=.6,slack=2.,penalty=.05):
    """Offline dynamic programming: retain a high-confidence, motion-consistent path.

    This uses the complete clip. It is not a zero-latency filter or a physical
    guarantee; speed is in relative mat-width percentage per source frame.
    """
    out=np.full_like(z,np.nan)
    for j in range(z.shape[1]):
        kk=np.flatnonzero(np.isfinite(z[:,j,0]));n=len(kk)
        if not n:continue
        xy=z[kk,j];score=np.zeros(n);previous=np.full(n,-1,int)
        for i in range(n):
            reward=.5+.5*confidence[kk[i],j];score[i]=reward
            if not i:continue
            dt=kk[i]-kk[:i];distance=np.linalg.norm(xy[:i]-xy[i],axis=1);valid=distance<=slack+max_speed*dt
            value=score[:i]+reward-penalty*distance**2/np.maximum(dt,1);value[~valid]=-np.inf;b=int(np.argmax(value))
            if value[b]>score[i]:score[i]=value[b];previous[i]=b
        i=int(np.argmax(score))
        while i>=0:out[kk[i],j]=xy[i];i=previous[i]
    return out

def fill_estimates(a,state,ages,cfg):
    """Bounded offline interpolation and short endpoint holds, always marked estimated."""
    for j in range(a.shape[1]):
        kk=np.flatnonzero(np.isfinite(a[:,j,0]))
        if not len(kk):continue
        for left,right in zip(kk,kk[1:]):
            gap=right-left-1
            if 0<gap<=cfg.get('fill_gap',0) and np.linalg.norm(a[right,j]-a[left,j])<=3+cfg.get('fill_speed',.3)*(right-left):
                for k in range(left+1,right):a[k,j]=a[left,j]+(a[right,j]-a[left,j])*(k-left)/(right-left);state[k,j]=2;ages[k,j]=min(k-left,right-k)
        hold=cfg.get('endpoint_hold',0)
        for k in range(max(0,kk[0]-hold),kk[0]):a[k,j]=a[kk[0],j];state[k,j]=2;ages[k,j]=kk[0]-k
        for k in range(kk[-1]+1,min(len(a),kk[-1]+hold+1)):a[k,j]=a[kk[-1],j];state[k,j]=2;ages[k,j]=k-kk[-1]
    return a,state,ages

def filtered(z,confidence,cfg):
    n,m,_=z.shape;res=np.full_like(z,np.nan);state=np.zeros((n,m),int);ages=np.full((n,m),999,int)
    zz=isolated_outliers(z,cfg.get('spike_limit',3)) if cfg.get('despike') else z.copy()
    if cfg.get('hampel'):zz=hampel_outliers(zz,cfg.get('hampel_window',9),cfg.get('hampel_limit',3))
    if cfg.get('path_gate'):zz=path_outliers(zz,confidence,cfg.get('path_speed',.6),cfg.get('path_slack',2.),cfg.get('path_penalty',.05))
    if cfg['kind']=='local':
        half=cfg['window']//2
        for j in range(m):
            valid=np.flatnonzero(np.isfinite(zz[:,j,0]));segments=np.split(valid,np.flatnonzero(np.diff(valid)>cfg.get('gap',8))+1)
            for segment in segments:
                if not len(segment):continue
                for k in range(segment[0],segment[-1]+1):
                    ix=segment[np.abs(segment-k)<=half]
                    if not len(ix):continue
                    nearest=int(np.min(np.abs(ix-k)))
                    if nearest>cfg.get('gap',8):continue
                    tt=(ix-k)/25.;degree=min(cfg.get('order',2),len(ix)-1);A=np.stack([tt**p for p in range(degree+1)],axis=1)
                    w=np.exp(-.5*((ix-k)/max(half*.65,1))**2)*np.maximum(confidence[ix,j],.05)
                    Y=zz[ix,j]
                    for step in range(cfg.get('robust',2)+1):
                        sw=np.sqrt(w);coef=np.linalg.lstsq(A*sw[:,None],Y*sw[:,None],rcond=None)[0]
                        if step<cfg.get('robust',2):
                            err=np.linalg.norm(Y-A@coef,axis=1);scale=max(.15,1.4826*np.median(np.abs(err-np.median(err))));w*=np.minimum(1,1.5*scale/np.maximum(err,1e-6))
                    if inside(coef[0]):
                        res[k,j]=coef[0];state[k,j]=1 if np.isfinite(zz[k,j]).all() else 2;ages[k,j]=nearest
        return fill_estimates(res,state,ages,cfg) if cfg.get('fill_gap') or cfg.get('endpoint_hold') else (res,state,ages)
    for j in range(m):
        p=None;last=-999;history=[];pending=[];derivative=np.zeros(2);prev_raw=None;prev_raw_k=None
        for k in range(n):
            v=zz[k,j];valid=np.isfinite(v).all()
            if valid and cfg.get('gate_raw') and p is not None and k-last<=cfg.get('hold',25):
                gate=cfg['gate_raw']+cfg.get('allowed_speed',.6)*max(k-last-1,0)
                if np.linalg.norm(v-p)>gate:
                    valid=False
            if valid:
                if k-last>cfg.get('hold',25):history=[];p=None;prev_raw=None
                history.append(v.copy());history=history[-cfg.get('median',1):];v=np.median(history,axis=0)
                if cfg.get('gate') and p is not None and np.linalg.norm(v-p)>cfg['gate']:
                    if pending and np.linalg.norm(v-pending[-1])>1.5:pending=[]
                    pending.append(v.copy());valid=len(pending)>=cfg.get('confirm',3)
                else:pending=[]
            if valid:
                if p is None:p=v.copy()
                elif cfg['kind']=='raw':p=v.copy()
                elif cfg['kind']=='euro':
                    dt=(k-prev_raw_k)/25 if prev_raw_k is not None else .04
                    dv=(v-prev_raw)/max(dt,.04) if prev_raw is not None else np.zeros(2)
                    ad=1/(1+1/(2*np.pi*cfg.get('dcutoff',1)*dt));derivative+=ad*(dv-derivative)
                    cutoff=cfg['cutoff']+cfg['beta']*np.linalg.norm(derivative);alpha=1/(1+1/(2*np.pi*cutoff*dt));p+=alpha*(v-p)
                elif cfg['kind']=='ema':
                    alpha=cfg['alpha']
                    if cfg.get('confidence_weighted'):alpha*=.4+.6*confidence[k,j]
                    delta=v-p
                    if cfg.get('step_cap'):delta*=min(1,cfg['step_cap']/max(np.linalg.norm(delta),1e-9))
                    p+=alpha*delta
                prev_raw=v.copy();prev_raw_k=k;last=k;res[k,j]=p;state[k,j]=1;ages[k,j]=0
            elif p is not None and k-last<=cfg.get('hold',25):res[k,j]=p;state[k,j]=2;ages[k,j]=k-last
    return res,state,ages

def references(base,labels,frames,ids):
    idx={v:i for i,v in enumerate(ids)};records=[]
    for k in frames:
        gt=[o for o in labels[str(k)] if o.get('box') is not None];rr=[o for o in base['frames'][k] if o['status']=='observed'];links=match_boxes([o['box'] for o in gt],[o['box'] for o in rr],.5)
        for i,o in enumerate(gt):records.append({'frame':k,'reference_id':o['id'],'track_index':idx[rr[links[i]]['id']] if i in links else None,'point':stage(o['floor']) if o['floor'] is not None else None,'posture':o['support_type'],'contact':o['contact']})
    return records

def scores(a,states,refs,frames):
    errors=[];visible=correct=matched=hidden_measured=hidden_estimated=0;bytype={}
    for r in refs:
        k=r['frame'];j=r['track_index'];p=a[k,j] if j is not None else np.array([np.nan,np.nan]);valid=np.isfinite(p).all()
        if r['point'] is None:
            if valid:hidden_measured+=states[k,j]==1;hidden_estimated+=states[k,j]==2
            continue
        visible+=1;t=r['posture'];bytype.setdefault(t,{'refs':0,'correct':0,'available':0});bytype[t]['refs']+=1
        if valid:
            e=float(np.linalg.norm(p-r['point']));errors.append(e);matched+=1;correct+=e<=3;bytype[t]['available']+=1;bytype[t]['correct']+=int(e<=3)
    emitted=int(np.isfinite(a[frames,:,0]).sum());measured=int((states[frames]==1).sum())
    return {'references':visible,'correct_within_3pct':int(correct),'available_matched':matched,'emitted':emitted,'measured_emitted':measured,'estimated_emitted':emitted-measured,'joint_f1':2*correct/max(visible+emitted,1),'median_error_pct':float(np.median(errors)) if errors else None,'p90_error_pct':float(np.percentile(errors,90)) if errors else None,'hidden_or_uncertain_measured':int(hidden_measured),'hidden_or_uncertain_estimated':int(hidden_estimated),'by_posture':bytype}

def continuity(a,b,common=None):
    mask=np.isfinite(a[:,:,0])&np.isfinite(b[:,:,0])
    if common is not None:mask&=common
    both=mask[1:]&mask[:-1];triple=mask[2:]&mask[1:-1]&mask[:-2];out={}
    for name,v in [('baseline',b),('candidate',a)]:
        step=np.linalg.norm(np.diff(v,axis=0),axis=2)[both];acc=np.linalg.norm(v[2:]-2*v[1:-1]+v[:-2],axis=2)[triple]
        out[name]={'paired_transitions':len(step),'step_p99':float(np.percentile(step,99)) if len(step) else None,'acceleration_p99':float(np.percentile(acc,99)) if len(acc) else None,'steps_over_1pct':int((step>1).sum()),'abrupt_changes_over_half_pct':int((acc>.5).sum())}
    return out

def stillness(a,b,labels,base,ids):
    # Identical sampled person/frame pairs for both methods. This is a short diagnostic.
    refs=references(base,labels,[285,299],ids);g0={r['reference_id']:r for r in refs if r['frame']==285};g1={r['reference_id']:r for r in refs if r['frame']==299};result=[]
    for i,r in g0.items():
        s=g1.get(i)
        if not s or r['point'] is None or s['point'] is None or r['track_index'] is None or r['track_index']!=s['track_index']:continue
        j=r['track_index'];kk=np.arange(285,300);mask=np.isfinite(a[kk,j,0])&np.isfinite(b[kk,j,0]);kk=kk[mask]
        if len(kk)<12:continue
        ref=r['point'][None]+(kk[:,None]-285)/14*(s['point']-r['point']);row={'reference_id':i,'pairs':len(kk)}
        for name,v in [('baseline',b),('candidate',a)]:
            residual=v[kk,j]-ref;residual-=residual.mean(axis=0);row[name]=float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
        result.append(row)
    return {'scope':'Final 0.56 seconds; fluctuation about interpolated reference endpoints, including held estimates. Shared pairs, at least12 per ID. Not absolute accuracy or dense motion truth.','rows':result,'baseline_median':float(np.median([r['baseline'] for r in result])) if result else None,'candidate_median':float(np.median([r['candidate'] for r in result])) if result else None}

def configs():
    yield {'name':'raw','kind':'raw','hold':0}
    for median,alpha in itertools.product([1,3,5],[.15,.25,.35,.5,.7]):
        yield {'name':f'ema_m{median}_a{alpha}','kind':'ema','median':median,'alpha':alpha,'hold':25}
    yield {'name':'old_filter','kind':'ema','median':3,'alpha':.35,'gate':2,'confirm':3,'step_cap':2.5,'hold':25}
    for cutoff,beta,median in itertools.product([.3,.6,1.2],[.03,.1,.3],[1,3]):
        yield {'name':f'euro_c{cutoff}_b{beta}_m{median}','kind':'euro','cutoff':cutoff,'beta':beta,'median':median,'hold':25}
    for window,order,despike in itertools.product([7,11,17,25],[1,2],[False,True]):
        yield {'name':f'local_w{window}_p{order}_d{int(despike)}','kind':'local','window':window,'order':order,'despike':despike,'robust':2,'gap':8}
    for gate,alpha,median in itertools.product([3.,5.,8.],[.35,.55],[1,3]):
        yield {'name':f'innovation_g{gate}_a{alpha}_m{median}','kind':'ema','gate_raw':gate,'allowed_speed':.6,'median':median,'alpha':alpha,'step_cap':2.,'hold':25}
    for window,order,gap in itertools.product([11,17,25],[1,2],[8,25]):
        yield {'name':f'hampel_local_w{window}_p{order}_gap{gap}','kind':'local','window':window,'order':order,'hampel':True,'hampel_window':9,'hampel_limit':3.,'robust':2,'gap':gap}
    for speed,window in itertools.product([.3,.6,1.],[11,17,25]):
        yield {'name':f'path_v{speed}_w{window}','kind':'local','window':window,'order':1,'path_gate':True,'path_speed':speed,'path_slack':2.,'path_penalty':.05,'robust':2,'gap':25,'lookahead':'Full clip for path selection; local window for smoothing'}
    for window,fill in itertools.product([17,25],[25,50,75]):
        yield {'name':f'path_filled_w{window}_gap{fill}','kind':'local','window':window,'order':1,'path_gate':True,'path_speed':.3,'path_slack':2.,'path_penalty':.05,'robust':2,'gap':25,'fill_gap':fill,'fill_speed':.3,'endpoint_hold':25,'lookahead':'Full clip; missing positions interpolated for bounded gaps and explicitly estimated'}

def export(base,a,states,ages,z,ids,cfg):
    d=copy.deepcopy(base);idx={v:i for i,v in enumerate(ids)}
    for k,row in enumerate(d['frames']):
        for o in row:
            j=idx[o['id']];v=a[k,j];o.update(map=v.tolist() if np.isfinite(v).all() else None,map_status={0:'unknown',1:'proxy',2:'estimated'}[int(states[k,j])],map_age_frames=int(ages[k,j]))
            for key in ['raw_map','uncertainty','point']:o.pop(key,None)
            if np.isfinite(z[k,j]).all():o['raw_map']=z[k,j].tolist()
            if states[k,j]==2:o['uncertainty']=.5+.1*ages[k,j]
    d['map_filter']=cfg;return d

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='v1_balanced_s29');parser.add_argument('--threshold',type=float);args=parser.parse_args();folder=OUT/'runs'/args.run
    base=read(ROOT/'Dance_Tracking_Pilot/tracks.json')['roster'];pred=read(folder/'controlled_predictions.json');threshold=args.threshold if args.threshold is not None else read(folder/'controlled_metrics.json')['selected']['threshold']
    dataset=read(OUT/'data/development_v1.json')['controlled'];labels=dataset['labels'];frames=dataset['val'];z,confidence,ids=inputs(base,pred,threshold);before,bstate=original_arrays(base,ids);refs=references(base,labels,frames,ids)
    results=[];out=folder/'filters';out.mkdir(exist_ok=True)
    for cfg in configs():
        a,st,age=filtered(z,confidence,cfg);r={'settings':cfg,'accuracy':scores(a,st,refs,frames),'continuity':continuity(a,before),'stationary':stillness(a,before,labels,base,ids)};results.append(r)
    baseline=scores(before,bstate,refs,frames);maxcorrect=max(r['accuracy']['correct_within_3pct'] for r in results)
    # Filter selection is explicit development tuning. Accuracy takes priority over cosmetic smoothness.
    eligible=[r for r in results if r['accuracy']['correct_within_3pct']>=maxcorrect-1 and r['accuracy']['hidden_or_uncertain_measured']<=results[0]['accuracy']['hidden_or_uncertain_measured']+1]
    chosen=min(eligible,key=lambda r:(r['continuity']['candidate']['acceleration_p99'],r['stationary']['candidate_median'] or 1e9))
    a,st,age=filtered(z,confidence,chosen['settings']);write(out/'selection.json',{'scope':'Development tuning, not independent test evidence.','threshold':threshold,'baseline':baseline,'rule':'Within1 of the highest correct-floor count; no more than1 extra hidden measured point versus raw. Then minimize acceleration p99, tie-break short stationary RMS.','chosen':chosen,'candidates':results});write(out/'tracks.json',export(base,a,st,age,z,ids,chosen['settings']))
    print(args.run,'baseline',baseline['correct_within_3pct'],'selected',chosen['settings']['name'],chosen['accuracy'],flush=True)

if __name__=='__main__':main()
