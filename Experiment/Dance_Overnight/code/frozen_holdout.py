"""Apply the development-frozen pipeline once to a temporally reserved section."""
import copy,hashlib,datetime,argparse,time
from pipeline_infer import infer,read,write,ROOT,OUT,torch,cv2,np
from candidate_evaluate import load_model,predict_aug
from development_search import ensemble
from filter_lab import inputs,filtered,export,original_arrays,continuity
from compare import stage,pixel,bottom,match_boxes
from flow_tracker import run as run_flow
from tracker_lab import color_feature
from detector_probe import candidates,merge,YOLO
from bidirectional_tracking import combine
from pipeline_eval import identity_score

LONG=OUT/'long_sequence';D=OUT/'data/holdout';DEST=OUT/'reserved_evaluation'

def ensure_tracker(kind):
    if kind in ['flow','original']:return LONG/('flow_tracks.json' if kind=='flow' else 'reference_tracks.json')
    path=LONG/f'{kind}_tracks.json'
    if path.exists():return path
    seeds=read(ROOT/'Dance_Corrected_Pilot/labels.json')['0'];n=661
    if kind=='frozen':
        cfg=read(OUT/'detector/sequences/posture_frozen_960/tracking_selection.json');folder=LONG/'frozen';folder.mkdir(exist_ok=True);cache=folder/'raw.json'
        pred=read(cache) if cache.exists() else read(OUT/'detector/sequences/posture_frozen_960/raw.json');oldfeat=read(OUT/'detector/sequences/posture_frozen_960/color_features.json');ff={'seeds':oldfeat['seeds'],'frames':{}}
        model=YOLO(OUT/'detector/runs/posture_frozen/weights/best.pt');cap=cv2.VideoCapture(str(LONG/'source.avi'))
        for k in range(n):
            ok,im=cap.read();assert ok
            if str(k) not in pred:pred[str(k)]=candidates(model,im,960,'0')
            if k%100==0:print('Frozen detector extension',k,flush=True);write(cache,pred)
            row=merge(pred[str(k)],cfg['detector_merge_iou']);ff['frames'][str(k)]=[color_feature(im,o['box']).tolist() for o in row]
        cap.release();write(cache,pred);merged={k:merge(row,cfg['detector_merge_iou']) for k,row in pred.items()};write(folder/'merged.json',merged);write(folder/'features.json',ff)
        tracks=run_flow(merged,seeds,ff,cfg['config'],np.load(LONG/'dense_flow.npz')['flow'].astype(np.float32));write(path,tracks)
    elif kind=='bidir':
        forward=read(LONG/'flow_tracks.json');pred=read(LONG/'predictions.json');ff=read(LONG/'color_features.json');reverse_seeds=[o for o in forward['frames'][-1] if o['status']=='observed'];cap=cv2.VideoCapture(str(LONG/'source.avi'));flow=[];prev=None;alg=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        for k in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES,n-1-k);ok,im=cap.read();assert ok
            if not k:seedf=[color_feature(im,o['box']).tolist() for o in reverse_seeds]
            gray=cv2.resize(cv2.cvtColor(im[280:600,100:1220],cv2.COLOR_BGR2GRAY),(560,160));flow.append(np.zeros((160,560,2),np.float32) if prev is None else alg.calc(prev,gray,None));prev=gray
        cap.release();rp={str(k):pred[str(n-1-k)] for k in range(n)};rf={'seeds':seedf,'frames':{str(k):ff['frames'][str(n-1-k)] for k in range(n)}}
        backward=run_flow(rp,reverse_seeds,rf,read(OUT/'tracking/flow_selection.json')['chosen']['config'],np.array(flow));write(LONG/'backward_tracks.json',backward);write(path,combine(forward,backward,'union'))
    else:raise ValueError(kind)
    return path

def predict_pipeline(cfg,base,folder,device='cuda'):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'predictions.json').exists():return read(folder/'predictions.json')
    if cfg.get('type')=='ensemble':
        pp=[predict_pipeline(c,base,folder/f'member{i}',device) for i,c in enumerate(cfg['members'])];pred=ensemble(*pp,cfg['ensemble_mode']);write(folder/'predictions.json',pred);write(folder/'settings.json',cfg);write(folder/'base_tracks.json',base)
    else:infer(cfg,base,LONG/'source.avi',folder,device)
    return read(folder/'predictions.json')

def oracle_predict(cfg,labels,folder):
    folder.mkdir(parents=True,exist_ok=True);path=folder/'oracle_predictions.json'
    if path.exists():return read(path)
    if cfg.get('type')=='ensemble':
        aa=oracle_predict(cfg['members'][0],labels,folder/'oracle_a');bb=oracle_predict(cfg['members'][1],labels,folder/'oracle_b');pred={k:ensemble([aa[k]],[bb[k]],cfg['ensemble_mode'])[0] for k in aa}
    else:
        net,mask,_=load_model(OUT/'runs'/cfg['run']/cfg['checkpoint']);pred={}
        for k,rows in labels.items():
            im=cv2.imread(str(D/'images'/f'{k}.png'));pp=predict_aug(net,im,rows,cfg['mode'],mask)
            if cfg['decoder']!='point':
                for p in pp.values():p['point']=p[cfg['decoder']]
            pred[k]=pp
        del net;torch.cuda.empty_cache()
    write(path,pred);return read(path)

def aggregate(records):
    visible=[r for r in records if r['contact']=='visible'];available=[r for r in visible if r['error_pct'] is not None];known=[r for r in records if r['contact']!='uncertain'];errors=[r['error_pct'] for r in available];counts={}
    for kind in sorted({r['kind'] for r in records}):
        rr=[r for r in records if r['kind']==kind];counts[kind]={'references':len(rr),'measured':sum(r['state']==1 for r in rr),'estimated':sum(r['state']==2 for r in rr),'unknown':sum(r['state']==0 for r in rr),'correct_3pct':sum(r['error_pct'] is not None and r['error_pct']<=3 for r in rr)}
    return {'visible_references':len(visible),'matched_visible_boxes':sum(r['matched'] for r in visible),'available_visible':len(available),'correct_by_tolerance_pct_width':{str(t):sum(e<=t for e in errors) for t in [1,2,3,5]},'median_error_pct':float(np.median(errors)) if errors else None,'p90_error_pct':float(np.percentile(errors,90)) if errors else None,'median_source_pixel_error':float(np.median([r['error_pixels'] for r in available])) if available else None,'hidden_references':sum(r['contact']=='hidden' for r in records),'hidden_measured':sum(r['contact']=='hidden' and r['state']==1 for r in records),'hidden_estimated':sum(r['contact']=='hidden' and r['state']==2 for r in records),'measured_correct_3pct':sum(r['state']==1 and r['error_pct'] is not None and r['error_pct']<=3 for r in visible),'by_posture':counts}

def result(records):
    phases=read(D/'sealed_protocol.json')['evaluation_precommit']['phases'];return {'all':aggregate(records),'by_phase':{k:aggregate([r for r in records if r['source_frame'] in frames]) for k,frames in phases.items()},'by_frame':{str(k):aggregate([r for r in records if r['source_frame']==k]) for k in sorted({r['source_frame'] for r in records})},'records':records}

def record(k,o,p,state,matched,track_id=None):
    valid=p is not None;gt=o['floor'];e=float(np.linalg.norm(np.array(p)-stage(gt))) if valid and gt is not None else None
    return {'source_frame':int(k),'reference_id':o['id'],'track_id':track_id,'contact':o['contact'],'kind':o['support_type'] or o['contact'],'matched':matched,'state':state if valid else 0,'error_pct':e,'error_pixels':float(np.linalg.norm(pixel(p)-np.array(gt))) if e is not None else None}

def oracle_score(pred,labels,threshold,box=False):
    rr=[]
    for k,rows in labels.items():
        for o in rows:
            if box:p=stage(bottom(np.array(o['box'])));valid=True
            else:
                q=pred[k][str(o['id'])];p=stage(q['point']);valid=q['reportability']>=threshold and 0<=p[0]<=100 and 0<=p[1]<=62.5
            rr.append(record(k,o,p if valid else None,1,True))
    return result(rr)

def track_score(data,labels,iou=.5,observed_only=False):
    rr=[]
    for k,gt in labels.items():
        pp=[o for o in data['frames'][int(k)-4420] if o['status']=='observed' or (not observed_only and o['status']=='predicted' and o['age_frames']<=25)]
        links=match_boxes([o['box'] for o in gt],[o['box'] for o in pp],iou)
        for i,o in enumerate(gt):
            q=pp[links[i]] if i in links else None;state={'proxy':1,'estimated':2,'unknown':0}[q['map_status']] if q else 0
            rr.append(record(k,o,q['map'] if q else None,state,q is not None,q['id'] if q else None))
    return result(rr)

def identity_audit(data):
    labels=read(D/'limited_identity_labels.json');frames=sorted(map(int,labels));ids=sorted({o['id'] for rows in data['frames'] for o in rows});a,st=original_arrays(data,ids);matched=correct=0
    for k in frames:
        gt=labels[str(k)];pp=[o for o in data['frames'][k] if o['status']=='observed'];links=match_boxes([o['box'] for o in gt],[o['box'] for o in pp],.5);matched+=len(links);correct+=sum(gt[i]['id']==pp[j]['id'] for i,j in links.items())
    return {'protocol':read(D/'limited_identity_protocol.json'),'box_references':80,'matched_observed_boxes':matched,'matches_with_correct_initial_id':correct,'wrong_initial_id_matches':matched-correct,'identity_aware_floor':identity_score(a,st,labels,frames,ids)}

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);DEST.mkdir(exist_ok=True);selection=read(OUT/'search/frozen_selection.json');sealed=read(D/'sealed_protocol.json');assert hashlib.sha256((D/'labels.json').read_bytes()).hexdigest()==sealed['label_sha256'];labels=read(D/'labels.json')
    write(DEST/'evaluation_started.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_sha256':hashlib.sha256((OUT/'search/frozen_selection.json').read_bytes()).hexdigest(),'label_sha256':sealed['label_sha256'],'scope':'First application of development-frozen pipelines to reserved frames. No threshold/filter tuning here.'})
    baseline=read(LONG/'reference_tracks.json');results={'protocol':sealed,'frozen_selection':selection,'baseline':track_score(baseline,labels),'baseline_limited_identity':identity_audit(baseline),'oracle_box':oracle_score(None,labels,0,True),'candidates':{}}
    for mode in ['causal','offline']:
        chosen=selection[mode]['chosen'];folder=DEST/mode;cfg=chosen['support'];base=read(ensure_tracker(chosen['tracker']));pred=predict_pipeline(cfg,base,folder);z,cf,ids=inputs(base,pred,chosen['threshold']);a,st,age=filtered(z,cf,chosen['settings']);tracks=export(base,a,st,age,z,ids,chosen['settings']);write(folder/'tracks.json',tracks)
        oracle=oracle_predict(cfg,labels,folder);ba,_=original_arrays(baseline,ids)
        results['candidates'][mode]={'end_to_end':track_score(tracks,labels),'observed_only':track_score(tracks,labels,observed_only=True),'iou03_sensitivity':track_score(tracks,labels,.3),'limited_identity':identity_audit(tracks),'oracle_crops':oracle_score(oracle,labels,chosen['threshold']),'continuity':continuity(a[380:],ba[380:]),'selection_name':chosen['name'],'threshold':chosen['threshold'],'filter':chosen['settings'],'track_config':base['config']}
        write(DEST/'results.json',results);print('RESERVED',mode,results['candidates'][mode]['end_to_end']['all'],flush=True)
    print('Reserved evaluation complete; settings unchanged',flush=True)

if __name__=='__main__':main()
