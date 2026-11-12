"""Select thresholds on validation, then evaluate frozen settings on the test block."""
import json,time
from collections import defaultdict
from types import SimpleNamespace
import cv2,numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from corrected_setup import OUT,SOURCE,TILES,VAL,TEST,read,get_image
from infer import WEIGHTS
from compare import match_boxes,overlap,bottom,stage,smooth

def predict(model,im):
    candidates=[]
    for (l,t,r,b),result in zip(TILES,model.predict([im[t:b,l:r] for l,t,r,b in TILES],imgsz=960,conf=.06,classes=[0],device=0,verbose=False)):
        for j,row in enumerate(result.boxes.data.cpu().numpy()):
            a=row.astype(float);a[:4]+=[l,t,l,t]
            if (l>0 and a[0]<l+5) or (r<1280 and a[2]>r-5):continue
            obj={'box':a[:4].tolist(),'score':float(a[4])}
            if result.keypoints is not None:
                p=result.keypoints.data[j,0].cpu().numpy()
                obj.update(point=(p[:2]+[l,t]).tolist(),point_conf=float(p[2]))
            candidates.append(obj)
    keep=[]
    for o in sorted(candidates,key=lambda o:o['score'],reverse=True):
        if not any(overlap(np.array(o['box']),np.array(q['box']))>.65 for q in keep):keep.append(o)
    return keep

def outputs(caches,k,mode,det_conf,point_conf=.5):
    which='baseline' if mode=='baseline' else 'pose' if mode=='pose' else 'detect'
    rows=[dict(o) for o in caches[which][str(k)] if o['score']>=det_conf]
    if mode=='hybrid':
        pp=[p for p in caches['pose'][str(k)] if p['score']>=.1]
        links=match_boxes([o['box'] for o in rows],[o['box'] for o in pp],.30)
        for i,j in links.items():rows[i].update(point=pp[j]['point'],point_conf=pp[j]['point_conf'])
    for o in rows:
        if mode in ('baseline','tuned_box'):o['point']=bottom(o['box']).tolist();o['point_conf']=1.
        if o.get('point_conf',0)<point_conf:o.pop('point',None)
        if 'point' in o:
            p=stage(o['point'])
            if not(0<=p[0]<=100 and 0<=p[1]<=62.5):o.pop('point',None)
    return rows

def score(caches,frames,mode,det_conf,point_conf=.5):
    labels=read(OUT/'labels.json');true_boxes=pred_boxes=matched=refpoints=emitted=localized=hidden=hidden_emissions=0
    errors=[];pixelerrors=[];perframe=[]
    for k in frames:
        gt=[o for o in labels[str(k)] if not o.get('outside')];pred=outputs(caches,k,mode,det_conf,point_conf)
        links=match_boxes([o['box'] for o in gt],[o['box'] for o in pred],.5)
        true_boxes+=len(gt);pred_boxes+=len(pred);matched+=len(links)
        refpoints+=sum(o['floor'] is not None for o in gt);emitted+=sum('point' in o for o in pred)
        hidden+=sum(o['floor'] is None for o in gt)
        for i,j in links.items():
            if 'point' not in pred[j]:continue
            if gt[i]['floor'] is None:hidden_emissions+=1;continue
            e=float(np.linalg.norm(stage(pred[j]['point'])-stage(gt[i]['floor'])));errors.append(e)
            pixelerrors.append(float(np.linalg.norm(np.array(pred[j]['point'])-gt[i]['floor'])))
            localized+=e<=3
        perframe.append({'frame':k,'gt_visible':len(gt),'predicted':len(pred),'matched':len(links)})
    precision=matched/max(pred_boxes,1);recall=matched/max(true_boxes,1)
    return {'boxes':{'ground_truth':true_boxes,'predicted':pred_boxes,'matched_iou50':matched,'precision':precision,'recall':recall,'f1':2*matched/max(true_boxes+pred_boxes,1)},
      'floor':{'visible_references':refpoints,'emitted_points':emitted,'matched_visible_points':len(errors),'correct_within_3pct_mat_width':localized,
       'correct_fraction_of_all_visible_references':localized/max(refpoints,1),'precision_within_3pct':localized/max(emitted,1),'joint_f1':2*localized/max(refpoints+emitted,1),
       'median_error_pct_mat_width':float(np.median(errors)) if errors else None,'p90_error_pct_mat_width':float(np.percentile(errors,90)) if errors else None,
       'median_error_pixels':float(np.median(pixelerrors)) if pixelerrors else None,'hidden_reference_count':hidden,'points_emitted_on_matched_hidden_supports':hidden_emissions},'per_frame':perframe}

def models():return {'baseline':WEIGHTS/'yolo26m.pt','detect':Path(read(OUT/'training_detect.json')['weights']),'pose':Path(read(OUT/'training_pose.json')['weights'])}
from pathlib import Path

def select():
    caches={};timings={}
    for name,path in models().items():
        model=YOLO(path);started=time.perf_counter();cache={str(k):predict(model,get_image(k)) for k in VAL}
        caches[name]=cache;timings[name]=time.perf_counter()-started
    settings={};val_scores={}
    thresholds=[.1,.15,.25,.35,.45,.55,.65]
    for name,mode in [('baseline','baseline'),('detect','tuned_box'),('pose','pose')]:
        table=[(score(caches,VAL,mode,c),c) for c in thresholds]
        best,c=max(table,key=lambda x:x[0]['boxes']['f1'])
        settings[name]={'det_conf':c};val_scores[name+'_box_selection']=best
    for mode in ('baseline','tuned_box','pose','hybrid'):
        which='baseline' if mode=='baseline' else 'pose' if mode=='pose' else 'detect'
        c=settings[which]['det_conf']
        ps=[.25,.40,.55,.70,.85,.95] if mode in ('pose','hybrid') else [.5]
        table=[(score(caches,VAL,mode,c,p),p) for p in ps]
        best,p=max(table,key=lambda x:x[0]['floor']['joint_f1'])
        settings[mode]={'det_conf':c,'point_conf':p};val_scores[mode]=best
    selected=max(('tuned_box','pose','hybrid'),key=lambda m:val_scores[m]['floor']['joint_f1'])
    locked={'settings':settings,'selected_pipeline':selected,'validation_scores':val_scores,
      'selection_rule':'Detection thresholds maximize validation box F1. Point thresholds and final pipeline maximize validation joint floor F1 (within3% of mat width, penalizing misses and all emitted wrong points). No test scores used.',
      'weights':{k:str(v) for k,v in models().items()},'labels_sha256':read(OUT/'dataset_summary.json')['labels_sha256']}
    (OUT/'operating_points.json').write_text(json.dumps(locked,indent=2));(OUT/'validation_predictions.json').write_text(json.dumps(caches))
    print('LOCKED',selected,settings,flush=True)

def full_inference():
    assert (OUT/'operating_points.json').exists()
    caches={}
    for name,path in models().items():
        model=YOLO(path);cap=cv2.VideoCapture(str(SOURCE/'source.mp4'));cache={};t=time.perf_counter()
        for k in range(300):
            ok,im=cap.read();assert ok
            cache[str(k)]=predict(model,im)
            if k%75==0:print(name,k,flush=True)
        cap.release();caches[name]=cache
        (OUT/f'predictions_{name}.json').write_text(json.dumps(cache))
        (OUT/f'inference_{name}.json').write_text(json.dumps({'frames':300,'seconds':time.perf_counter()-t}))
    settings=read(OUT/'operating_points.json');result={}
    for mode in ('baseline','tuned_box','pose','hybrid'):
        result[mode]=score(caches,TEST,mode,**settings['settings'][mode])
    result['selected_using_validation']=settings['selected_pipeline']
    result['scope']='Eight reviewed frames in the final3seconds of the same video. Approximate assistant-reviewed labels; not an independent human audit or cross-video benchmark. No metres; fixed draft homography. Raw predictions evaluated before smoothing.'
    (OUT/'test_metrics.json').write_text(json.dumps(result,indent=2))
    tracking(caches,settings)
    stationary_check(caches,settings)
    print(json.dumps(result,indent=2),flush=True)

def tracking(caches,settings):
    labels=read(OUT/'labels.json');all_tracks={};temporal={}
    for mode in ('baseline','tuned_box','pose','hybrid'):
        cfg=settings['settings'][mode];tracker=BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.10,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True))
        seed=labels['0'];init=np.array([o['box']+[.99,0] for o in seed],np.float32)
        tracker.update(Boxes(init,(720,1280)));mapping={t.track_id:o['id'] for t,o in zip(tracker.tracked_stracks,seed)}
        rows=[]
        for k in range(300):
            # Keep low-confidence detections available for ByteTrack association.
            pred=outputs(caches,k,mode,.10,cfg['point_conf'])
            if k==0:active=[{'id':o['id'],'box':o['box']} for o in seed]
            else:
                arr=np.array([o['box']+[o['score'],0] for o in pred],np.float32).reshape(-1,6)
                tracked=tracker.update(Boxes(arr,(720,1280)))
                active=[]
                for a in tracked:
                    tid=int(a[4])
                    if tid not in mapping:mapping[tid]=101+len(mapping)-20
                    active.append({'id':mapping[tid],'box':a[:4].tolist()})
            links=match_boxes([o['box'] for o in active],[o['box'] for o in pred],.30)
            for i,j in links.items():
                if 'point' in pred[j]:active[i]['point']=pred[j]['point']
            if mode in ('baseline','tuned_box'):
                for o in active:o['point']=bottom(o['box']).tolist()
            rows.append(active)
        raw=defaultdict(dict)
        for k,row in enumerate(rows):
            for o in row:
                if 'point' in o:
                    p=stage(o['point'])
                    if 0<=p[0]<=100 and 0<=p[1]<=62.5:raw[o['id']][k]=p
        series={}
        for i,seq in raw.items():
            sm,filled=smooth(seq);series[i]={'raw':seq,'smooth':sm,'filled':sorted(filled)}
        observed=defaultdict(list)
        for k in TEST:
            gt=[o for o in labels[str(k)] if not o.get('outside')]
            links=match_boxes([o['box'] for o in gt],[o['box'] for o in rows[k]],.50)
            for i,j in links.items():observed[gt[i]['id']].append((k,rows[k][j]['id']))
        pairs=[(a,b) for seq in observed.values() for a,b in zip(seq,seq[1:]) if b[0]-a[0]<=20]
        temporal[mode]={'matched_test_reference_boxes':sum(len(v) for v in observed.values()),'reviewed_identity_pairs':len(pairs),'id_changes_between_reviewed_frames':sum(a[1]!=b[1] for a,b in pairs),'new_ids_full_clip':len({o['id'] for row in rows for o in row if o['id']>20}),'median_active_tracks_full_clip':float(np.median([len(r) for r in rows[1:]]))}
        all_tracks[mode]={'frames':rows,'series':series}
    def encode(o):return o.tolist() if isinstance(o,np.ndarray) else int(o) if isinstance(o,np.integer) else o
    (OUT/'tracks.json').write_text(json.dumps(all_tracks,default=encode));(OUT/'tracking_metrics.json').write_text(json.dumps(temporal,indent=2))
    (OUT/'tracking_settings.json').write_text(json.dumps({'track_high_thresh':.25,'track_low_thresh':.10,'new_track_thresh':.25,'track_buffer':30,'match_thresh':.8,'fuse_score':True,'scope':'Same tracker parameters for every model. New IDs renumbered compactly per method for display. Raw detection score tables separately use validation-selected confidence thresholds.'},indent=2))

def stationary_check(caches,settings):
    labels=read(OUT/'labels.json');ids=[4,5,6,7,9,10,11,15,16,17,20]
    start={o['id']:o for o in labels['285']};end={o['id']:o for o in labels['299']}
    modes=['baseline',settings['selected_pipeline']];res={m:defaultdict(list) for m in modes}
    for k in range(285,300):
        alpha=(k-285)/14;gt=[]
        for i in ids:
            if start[i]['floor'] is None or end[i]['floor'] is None:continue
            gt.append({'id':i,'box':(np.array(start[i]['box'])*(1-alpha)+np.array(end[i]['box'])*alpha).tolist(),'floor':np.array(start[i]['floor'])*(1-alpha)+np.array(end[i]['floor'])*alpha})
        for mode in modes:
            pred=outputs(caches,k,mode,**settings['settings'][mode]);links=match_boxes([o['box'] for o in gt],[o['box'] for o in pred],.5)
            for i,j in links.items():
                if 'point' in pred[j]:res[mode][gt[i]['id']].append((stage(pred[j]['point'])-stage(gt[i]['floor'])).tolist())
    common=[i for i in ids if all(len(res[m][i])>=12 for m in modes)]
    result={'window_frames':[285,299],'paired_dancer_ids':common,'minimum_observations_per_method':12,
      'scope':'Approximate stationary kneeling-support check over final0.56seconds. Reference positions and boxes interpolated only for evaluation between reviewed endpoints. Raw point residuals have their mean removed; measures local fluctuation, not full-video jitter or absolute accuracy.'}
    for m in modes:
        vals={i:float(np.sqrt(np.mean(np.sum((np.array(res[m][i])-np.mean(res[m][i],axis=0))**2,axis=1)))) for i in common}
        result[m]={'rms_fluctuation_pct_mat_width_by_id':vals,'median_rms_pct_mat_width':float(np.median(list(vals.values()))) if vals else None}
    (OUT/'stationary_check.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    import sys
    if sys.argv[1]=='select':select()
    elif sys.argv[1]=='infer':full_inference()
