"""Frozen-track pose experiment: replace box-bottom map input with ankle/support proposals."""
import copy,json,time,hashlib,sys
from pathlib import Path
from collections import deque,Counter
import cv2,numpy as np
from compare import stage,bottom,overlap,match_boxes
from corrected_setup import OUT as LABEL_DIR,SOURCE,VAL,TEST,TILES,read
from infer import WEIGHTS
from occlusion_pilot import OUT as TRACK_DIR,inside,evaluate
from map_kalman_pilot import frozen_hash,motion_stats,stationary,delay_check

OUT=TRACK_DIR.parent/'Dance_Ankle_Pilot';OUT.mkdir(exist_ok=True)
MODEL=WEIGHTS/'yolo26m-pose.pt'

def write(name,obj):
    def enc(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v))
    (OUT/name).write_text(json.dumps(obj,default=enc,indent=2),encoding='utf-8')

def infer():
    from ultralytics import YOLO
    model=YOLO(str(MODEL));tracks=read(TRACK_DIR/'tracks.json')['roster']['frames']
    cache_path=OUT/'pose_candidates.json';cache=read(cache_path) if cache_path.exists() else {}
    cap=cv2.VideoCapture(str(SOURCE/'source.mp4'));started=time.perf_counter();new=0
    for k in range(300):
        ok,im=cap.read();assert ok
        if str(k) in cache:continue
        regions=[(l,t,r,b,'tile') for l,t,r,b in TILES]
        for o in tracks[k]:
            if o['status']!='observed':continue
            x1,y1,x2,y2=o['box'];w=x2-x1;h=y2-y1
            px=max(14,w*.25);py=max(12,h*.15)
            regions.append((max(0,int(x1-px)),max(0,int(y1-py)),min(1280,int(x2+px)),min(720,int(y2+py)),f"crop_{o['id']}"))
        candidates=[]
        # Context tiles find complete bodies; enlarged fixed-track crops give small people more pixels.
        for start in range(0,len(regions),6):
            batch=regions[start:start+6];imgs=[im[t:b,l:r] for l,t,r,b,_ in batch]
            results=model.predict(imgs,imgsz=768,conf=.10,classes=[0],device=0,quantize=16,verbose=False)
            for (l,t,r,b,source),res in zip(batch,results):
                boxes=res.boxes.data.cpu().numpy();kp=res.keypoints.data.cpu().numpy()
                assert kp.shape[1]==17,'Must use pretrained COCO pose, not the one-point contact model.'
                for box,points in zip(boxes,kp):
                    bb=box[:4]+[l,t,l,t];pp=points.copy();pp[:,:2]+=[l,t]
                    # Truncated crop detections cannot establish the full person's ankle location.
                    if (l>0 and bb[0]<l+2) or (r<1280 and bb[2]>r-2):continue
                    candidates.append({'box':bb.tolist(),'score':float(box[4]),'kp':pp.tolist(),'source':source})
        keep=[]
        for o in sorted(candidates,key=lambda o:o['score'],reverse=True):
            if not any(overlap(np.array(o['box']),np.array(q['box']))>.65 for q in keep):keep.append(o)
        cache[str(k)]=keep;new+=1
        if k%25==0 or k==299:
            write('pose_candidates.json',cache)
            print(f'Pose frame {k+1}/300; {len(keep)} candidates; {time.perf_counter()-started:.1f}s',flush=True)
    cap.release();write('inference.json',{'model':str(MODEL),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'new_frames':new,'seconds':time.perf_counter()-started,'input':'Three context tiles and padded crops from frozen observed track boxes. Batch6, image size768, half precision, GPU0. Keypoints are COCO17: knees13/14, ankles15/16.','keypoint_confidence_is_not_contact_probability':True})

def associate(base,poses):
    rows=[]
    for k,rr in enumerate(base['frames']):
        active=[o for o in rr if o['status']=='observed'];pp=poses[str(k)]
        links=match_boxes([o['box'] for o in active],[o['box'] for o in pp],.35)
        mapping={active[i]['id']:{**pp[j],'association_iou':float(overlap(np.array(active[i]['box']),np.array(pp[j]['box'])))} for i,j in links.items()}
        rows.append(mapping)
    return rows

def angle(a,b,c):
    v=a-b;w=c-b;d=np.linalg.norm(v)*np.linalg.norm(w)
    return float(np.degrees(np.arccos(np.clip(v@w/max(d,1e-9),-1,1))))

def anchor(o,pose,cfg,others):
    if pose is None:return None,'no_matched_pose',[]
    kp=np.array(pose['kp']);threshold=cfg['confidence'];height=max(30,o['box'][3]-o['box'][1])
    ankles=[j for j in [15,16] if kp[j,2]>=threshold and kp[j,0]>0 and kp[j,1]>0]
    if not ankles:return None,'ankles_uncertain',[]
    selected=ankles;kind='ankle_pair' if len(ankles)==2 else 'single_ankle'
    if cfg['mode']!='midpoint' and len(ankles)==2:
        # Reject an obviously raised ankle: above the opposite knee AND substantially
        # above the opposite ankle. Image geometry is only a contact heuristic.
        raised=[]
        for ankle,knee,other in [(15,14,16),(16,13,15)]:
            if kp[knee,2]>=threshold and kp[ankle,1]<kp[knee,1] and kp[other,1]-kp[ankle,1]>.12*height:raised.append(ankle)
        selected=[j for j in ankles if j not in raised]
        if not selected:return None,'raised_or_ambiguous_ankles',[]
        if len(selected)==1:kind='lower_support_ankle'
    if cfg['mode']=='support' and len(ankles)==2 and all(kp[j,2]>=threshold for j in [11,12,13,14]):
        bends=[angle(kp[h,:2],kp[n,:2],kp[a,:2]) for h,n,a in [(11,13,15),(12,14,16)]]
        below=[kp[n,1]-kp[a,1] for n,a in [(13,15),(14,16)]]
        if all(v<120 for v in bends) and all(v>.035*height for v in below):
            selected=[13,14];kind='kneeling_knee_proposal'
    # A high joint score does not establish visibility; do not accept a support
    # landmark inside the body of a nearer observed dancer.
    visible=[]
    for j in selected:
        p=kp[j,:2];covered=False
        for other in others:
            if other['id']==o['id'] or other['status']!='observed':continue
            l,t,r,b=other['box']
            if l+3<p[0]<r-3 and t<p[1]<b-8:covered=True;break
        if not covered:visible.append(j)
    if not visible:return None,'landmarks_possibly_occluded',selected
    if len(visible)!=len(selected):kind='one_uncovered_'+kind
    point=kp[visible,:2].mean(axis=0)
    if not inside(stage(point)):return None,'landmarks_off_mat',visible
    return point,kind,visible

def maps(base,associations,cfg,anchor_fn=None):
    anchor_fn=anchor_fn or anchor
    out=copy.deepcopy(base);states={i:{'map':None,'last':-1000,'pending':[],'history':deque(maxlen=3)} for i in range(1,21)}
    for k,row in enumerate(out['frames']):
        original=base['frames'][k]
        for o in row:
            st=states[o['id']];pose=associations[k].get(o['id'])
            p,reason,used=anchor_fn(o,pose,cfg,original) if o['status']=='observed' else (None,'person_not_observed',[])
            for key in ['raw_map','uncertainty','point']:o.pop(key,None)
            o.update(anchor_reason=reason,used_landmarks=used,pose=pose,anchor_pixel=p)
            valid=p is not None
            if valid:
                raw=stage(p)
                if st['map'] is not None and k-st['last']<25 and np.linalg.norm(raw-st['map'])>2:
                    if st['pending'] and np.linalg.norm(raw-st['pending'][-1])>1.5:st['pending']=[]
                    st['pending'].append(raw.copy());valid=len(st['pending'])>=3
                    if not valid:o['anchor_reason']+=';jump_pending'
                else:st['pending']=[]
            if valid:
                st['history'].append(raw.copy());measurement=np.median(st['history'],axis=0)
                if st['map'] is None or k-st['last']>25:st['map']=measurement
                else:
                    v=measurement-st['map'];st['map']+=.35*v*min(1,2.5/max(np.linalg.norm(v),1e-9))
                st['last']=k;o.update(map=st['map'].copy(),map_status='proxy',raw_map=raw)
            elif st['map'] is not None and k-st['last']<=25:o.update(map=st['map'].copy(),map_status='estimated',uncertainty=.5+.10*(k-st['last']))
            else:o.update(map=None,map_status='unknown')
            o['map_age_frames']=k-st['last']
    out['anchor_settings']=cfg
    return out

def raw_metrics(data,labels,frames):
    refs=valid=correct=emitted=hidden=hidden_emissions=0;errors=[];kinds=Counter()
    for k in frames:
        gt=[o for o in labels[str(k)] if not o.get('outside')];rr=[o for o in data['frames'][k] if o['status']=='observed']
        links=match_boxes([o['box'] for o in gt],[o['box'] for o in rr],.5)
        refs+=sum(o['floor'] is not None for o in gt)
        emitted+=sum(o.get('anchor_pixel') is not None for o in rr)
        for o in rr:
            if o.get('anchor_pixel') is not None:kinds[o['anchor_reason'].split(';')[0]]+=1
        for i,j in links.items():
            g,o=gt[i],rr[j]
            if g['floor'] is None:
                hidden+=1;hidden_emissions+=o.get('anchor_pixel') is not None;continue
            if o.get('anchor_pixel') is None:continue
            e=float(np.linalg.norm(stage(o['anchor_pixel'])-stage(g['floor'])));errors.append(e);valid+=1;correct+=e<=3
    return {'visible_references':refs,'emitted_anchors':emitted,'matched_visible_anchors':valid,'correct_within_3pct':correct,'joint_f1':2*correct/max(refs+emitted,1),'median_error_pct_width':float(np.median(errors)) if errors else None,'matched_hidden_supports':hidden,'anchors_emitted_on_hidden_supports':hidden_emissions,'anchor_kinds':dict(kinds)}

def evaluate_all():
    base=read(TRACK_DIR/'tracks.json')['roster'];labels=read(LABEL_DIR/'labels.json');poses=read(OUT/'pose_candidates.json');assert len(poses)==300
    assoc=associate(base,poses);write('pose_associations.json',assoc)
    configs=[{'mode':m,'confidence':c} for m in ['midpoint','ankle_support','support'] for c in [.35,.55]]
    runs=[];results=[]
    for cfg in configs:
        data=maps(base,assoc,cfg);assert frozen_hash(data)==frozen_hash(base)
        results.append({'settings':cfg,'raw_validation':raw_metrics(data,labels,VAL),'map_validation':evaluate(data,labels,VAL)});runs.append(data)
    idx=max(range(len(results)),key=lambda i:(results[i]['raw_validation']['joint_f1'],results[i]['raw_validation']['correct_within_3pct']))
    after=runs[idx]
    write('selection.json',{'selected':idx,'rule':'Maximize raw support-point joint F1 on four existing validation frames, tolerance3% mat width; tie-break correct count. Penalizes missing references and all emitted points. Six fixed options: midpoint, raised-ankle rejection, raised-ankle rejection plus kneeling-knee heuristic; two joint confidence thresholds. All are proxies, not contact classifiers.','candidates':results})
    write('settings.json',configs[idx]);write('tracks.json',{'current':base,'ankles':after,'midpoint':runs[0]})
    raw_base=copy.deepcopy(base)
    for row in raw_base['frames']:
        for o in row:o['anchor_pixel']=bottom(np.array(o['box'])).tolist() if o['status']=='observed' and inside(stage(bottom(np.array(o['box'])))) else None;o['anchor_reason']='box_bottom'
    scores={split:{name:{'raw':raw_metrics(d,labels,frames),'map':evaluate(d,labels,frames)} for name,d in [('current',raw_base),('ankles',after)]} for split,frames in [('validation',VAL),('development_final',TEST)]}
    write('accuracy.json',scores);write('motion.json',motion_stats(base,after));write('stationary.json',stationary(base,after,labels))
    coverage={name:dict(Counter(o['map_status'] for row in d['frames'] for o in row)) for name,d in [('current',base),('ankles',after)]}
    write('coverage.json',coverage)
    write('provenance.json',{'frozen_boxes_ids_status_sha256':frozen_hash(base),'same_detector_boxes_ids_status':frozen_hash(base)==frozen_hash(after),'input_tracks_sha256':hashlib.sha256((TRACK_DIR/'tracks.json').read_bytes()).hexdigest(),'model':str(MODEL),'no_retraining':True,'source':str(SOURCE/'source.mp4'),'map_smoothing':'Same three-reading median, .35 bounded EMA,2%-jump gate with3-reading confirmation,25-frame maximum hold as current map. Only the input point and its availability change.','limits':['COCO17 has ankles, not heels/toes/sole contact.','Ankle confidence is not ground-contact confidence.','Raised-foot and kneeling checks are geometry heuristics.','No fallback to box bottoms in the ankle map.','Existing approximate support labels may use knees in floor work.','Already-reviewed development clip; no fresh independent holdout.']})
    print(json.dumps({'selected':configs[idx],'accuracy':json.loads((OUT/'accuracy.json').read_text())['development_final'],'coverage':coverage},indent=2),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='infer':infer()
    else:evaluate_all()
