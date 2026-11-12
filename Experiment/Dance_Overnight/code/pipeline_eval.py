"""Identity-aware end-to-end floor metrics and development filter selection."""
import argparse,itertools
from filter_lab import *

def identity_score(a,states,labels,frames,ids):
    idx={v:i for i,v in enumerate(ids)};errors=[];measured_errors=[];kinds={};counts={};records=[]
    for k in frames:
        for o in labels[str(k)]:
            j=idx.get(o['id']);p=a[k,j] if j is not None else np.array([np.nan,np.nan]);valid=np.isfinite(p).all();state=int(states[k,j]) if valid else 0;kind=o['support_type'] if o['floor'] is not None else o['contact'];counts.setdefault(kind,{'refs':0,'measured':0,'estimated':0,'unknown':0});counts[kind]['refs']+=1;counts[kind][{0:'unknown',1:'measured',2:'estimated'}[state]]+=1
            if o['floor'] is None:continue
            kinds.setdefault(kind,{'refs':0,'available':0,'correct_3pct':0});kinds[kind]['refs']+=1
            e=float(np.linalg.norm(p-stage(o['floor']))) if valid else None
            records.append({'frame':k,'id':o['id'],'kind':kind,'error':e,'state':state})
            if valid:
                errors.append(e);kinds[kind]['available']+=1;kinds[kind]['correct_3pct']+=int(e<=3)
                if state==1:measured_errors.append(e)
    refs=len(records);correct={str(t):sum(e<=t for e in errors) for t in [1,2,3,5]};measured_correct=sum(e<=3 for e in measured_errors);measured=sum(v['measured'] for k,v in counts.items() if k!='uncertain');hidden=sum(v['measured'] for k,v in counts.items() if k in ['hidden','airborne','outside'])
    return {'scope':'Same initial dancer ID required. Missing or wrong-ID outputs count as misses; held/interpolated outputs assessed separately. Uncertain labels excluded from visibility/F1.','visible_references':refs,'available_visible':len(errors),'correct_by_tolerance_pct_width':correct,'correct_measured_3pct':measured_correct,'measured_emitted_known_labels':measured,'measured_joint_f1':2*measured_correct/max(refs+measured,1),'hidden_measured':hidden,'median_error_pct':float(np.median(errors)) if errors else None,'p90_error_pct':float(np.percentile(errors,90)) if errors else None,'by_posture':kinds,'status_counts':counts,'records':records}

def dense_motion(a,ids):
    labels=read(OUT/'data/motion_review/labels.json');idx={v:i for i,v in enumerate(ids)};rows=[]
    for ident in [3,8,13,18]:
        pairs=[]
        for k in sorted(map(int,labels)):
            r=next(o for o in labels[str(k)] if o['id']==ident);q=a[k,idx[ident]]
            if np.isfinite(q).all():pairs.append((k,q,stage(r['floor'])))
        if len(pairs)<8:continue
        pred=np.array([x[1] for x in pairs]);gt=np.array([x[2] for x in pairs]);res=pred-gt
        rows.append({'id':ident,'pairs':len(pairs),'rmse_pct':float(np.sqrt(np.mean(np.sum(res**2,axis=1)))),'demeaned_rmse_pct':float(np.sqrt(np.mean(np.sum((res-res.mean(0))**2,axis=1)))),'reference_motion_rms_pct':float(np.sqrt(np.mean(np.sum((gt-gt.mean(0))**2,axis=1))))})
    return {'scope':'Four front-row identities on16 visually reviewed samples across1.2s. Small motion and several-pixel label uncertainty; this is not a reliable absolute latency benchmark.','rows':rows,'median_demeaned_rmse_pct':float(np.median([r['demeaned_rmse_pct'] for r in rows])) if rows else None}

def run(folder):
    folder=Path(folder);base=read(folder/'base_tracks.json');pred=read(folder/'predictions.json');dataset=read(OUT/'data/development_v1.json')['controlled'];labels=dataset['labels'];frames=dataset['val'];ids=sorted({o['id'] for row in base['frames'] for o in row});before,bstate=original_arrays(base,ids);thresholds=[]
    for t in [.1,.25,.4,.5,.6,.75,.9]:
        z,cf,_=inputs(base,pred,t);st=np.isfinite(z[:,:,0]).astype(int);s=identity_score(z,st,labels,frames,ids);thresholds.append({'threshold':t,'accuracy':s})
    selected=max(thresholds,key=lambda r:(r['accuracy']['measured_joint_f1'],r['accuracy']['correct_measured_3pct']));t=selected['threshold'];z,cf,ids=inputs(base,pred,t);results=[]
    for cfg in configs():
        a,st,age=filtered(z,cf,cfg);score=identity_score(a,st,labels,frames,ids);record={'settings':cfg,'accuracy':score,'continuity':continuity(a,before),'stationary':stillness(a,before,labels,base,ids),'dense_motion':dense_motion(a,ids)};results.append(record)
    # Compare a practical causal option and an offline option separately.
    chosen={};out=folder/'evaluation';out.mkdir(exist_ok=True)
    for mode in ['causal','offline']:
        family=[r for r in results if (r['settings']['kind']=='local')==(mode=='offline')]
        # A sparse-frame score must not hide a gross jump between those frames.
        pool=[r for r in family if r['continuity']['candidate']['step_p99']<=1. and all(x['demeaned_rmse_pct']<1. for x in r['dense_motion']['rows'])]
        if not pool:pool=family
        maxgood=max(r['accuracy']['correct_by_tolerance_pct_width']['3'] for r in pool);rawhidden=results[0]['accuracy']['hidden_measured'];eligible=[r for r in pool if r['accuracy']['correct_by_tolerance_pct_width']['3']>=maxgood-1 and r['accuracy']['hidden_measured']<=rawhidden+1]
        if not eligible:eligible=pool
        c=min(eligible,key=lambda r:(r['continuity']['candidate']['acceleration_p99'] or 0,r['dense_motion']['median_demeaned_rmse_pct'] or 1e9));chosen[mode]=c;a,st,age=filtered(z,cf,c['settings']);write(out/f'{mode}_tracks.json',export(base,a,st,age,z,ids,c['settings']))
    raw,rs,ra=filtered(z,cf,{'kind':'raw','hold':0});write(out/'raw_tracks.json',export(base,raw,rs,ra,z,ids,{'kind':'raw','hold':0}));old=read(ROOT/'Dance_Tracking_Pilot/tracks.json')['roster'];oa,os=original_arrays(old,ids)
    result={'scope':'Development tuning only; earlier pilot test frames are now development. Reserved4800..5080 not used.','selection':'Threshold maximizes identity-aware measured jointF1. Each causal/offline family first requires paired step p99<=1% mat width and every dense-motion dancer demeanedRMSE<1%, if feasible. Then within1 of maximum correct visible-floor count and no more than1 extra hidden measured contact; minimize paired acceleration p99. These thresholds are development choices, not physical speed guarantees. All settings retained.','threshold':t,'threshold_candidates':thresholds,'current_baseline':identity_score(oa,os,labels,frames,ids),'same_tracker_box_baseline':identity_score(before,bstate,labels,frames,ids),'chosen':chosen,'candidates':results};write(out/'results.json',result)
    print(folder.name,'threshold',t,'current',result['current_baseline']['correct_by_tolerance_pct_width'],'same tracker box',result['same_tracker_box_baseline']['correct_by_tolerance_pct_width'],flush=True)
    for mode,c in chosen.items():print(mode,c['settings'],c['accuracy']['correct_by_tolerance_pct_width'],'hidden measured',c['accuracy']['hidden_measured'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();run(a.folder)
