"""Actual toe/heel landmarks, using fixed YOLO boxes as top-down pose inputs."""
import sys,time,json,hashlib,copy
from pathlib import Path
from collections import Counter
import cv2,numpy as np
import ankle_pilot as ankle
from ankle_pilot import read,LABEL_DIR,SOURCE,TRACK_DIR,VAL,TEST,stage,inside,angle,maps,raw_metrics,evaluate,frozen_hash,motion_stats,stationary

OUT=TRACK_DIR.parent/'Dance_Feet_Pilot';OUT.mkdir(exist_ok=True)
URL='https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip'

def write(name,obj):
    def enc(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v))
    (OUT/name).write_text(json.dumps(obj,default=enc,indent=2),encoding='utf-8')

def infer():
    import torch,onnxruntime as ort
    ort.preload_dlls()
    from rtmlib import RTMPose
    model=RTMPose(URL,model_input_size=(288,384),to_openpose=False,backend='onnxruntime',device='cuda')
    print('Providers',model.session.get_providers(),flush=True)
    assert 'CUDAExecutionProvider' in model.session.get_providers(),'GPU provider must be active'
    base=read(TRACK_DIR/'tracks.json')['roster'];cache=read(OUT/'wholebody.json') if (OUT/'wholebody.json').exists() else {}
    cap=cv2.VideoCapture(str(SOURCE/'source.mp4'));start=time.perf_counter();new=0
    for k in range(300):
        ok,im=cap.read();assert ok
        if str(k) in cache:continue
        active=[o for o in base['frames'][k] if o['status']=='observed'];boxes=[o['box'] for o in active]
        rr={}
        if boxes:
            points,scores=model(im,bboxes=boxes)
            assert points.shape[1]==133 and len(points)==len(active)
            for o,pp,ss in zip(active,points,scores):
                rr[o['id']]={'box':o['box'],'score':o['score'],'kp':np.c_[pp[:23],ss[:23]].tolist(),'source':'RTMW_wholebody_topdown_frozen_box'}
        cache[str(k)]=rr;new+=1
        if k%25==0 or k==299:
            write('wholebody.json',cache);print(f'Feet frame {k+1}/300; {len(rr)} poses; {time.perf_counter()-start:.1f}s',flush=True)
    cap.release();write('inference.json',{'url':URL,'local_model':str(model.onnx_model),'model_sha256':hashlib.sha256(Path(model.onnx_model).read_bytes()).hexdigest(),'providers':model.session.get_providers(),'new_frames':new,'seconds':time.perf_counter()-start,'input_size':[288,384],'model':'RTMW whole-body133, retained first23 body+feet joints','association':'Direct top-down pose on each frozen observed YOLO box; no additional detector or ID changes.','foot_indices':{'left_big_toe':17,'left_small_toe':18,'left_heel':19,'right_big_toe':20,'right_small_toe':21,'right_heel':22}})

def contact(o,pose,cfg,others):
    if pose is None:return None,'no_pose',[]
    kp=np.array(pose['kp']);c=cfg['confidence'];height=max(30,o['box'][3]-o['box'][1]);feet=[]
    for ids in [[17,18,19],[20,21,22]]:
        valid=[j for j in ids if kp[j,2]>=c]
        feet.append(max(valid,key=lambda j:kp[j,1]) if valid else None)
    selected=[j for j in feet if j is not None];kind='toe_heel_support'
    if not selected:return None,'feet_uncertain',[]
    if len(selected)==2 and cfg['mode']!='feet_midpoint':
        l,r=selected
        if kp[14,2]>=c and kp[l,1]<kp[14,1]+.03*height and kp[r,1]-kp[l,1]>.10*height:selected=[r];kind='right_support_foot'
        elif kp[13,2]>=c and kp[r,1]<kp[13,1]+.03*height and kp[l,1]-kp[r,1]>.10*height:selected=[l];kind='left_support_foot'
    if cfg['mode']=='feet_knees' and all(kp[j,2]>=c for j in [11,12,13,14,15,16]):
        bends=[angle(kp[h,:2],kp[n,:2],kp[a,:2]) for h,n,a in [(11,13,15),(12,14,16)]]
        # Knees can be the support during floor work. Require both bent legs and
        # knees visibly below their ankle; still a proposal, not verified contact.
        if all(v<125 for v in bends) and all(kp[n,1]-kp[a,1]>.025*height for n,a in [(13,15),(14,16)]):selected=[13,14];kind='kneeling_knee_proposal'
    accepted=[]
    for j in selected:
        p=kp[j,:2];covered=any(q['id']!=o['id'] and q['status']=='observed' and q['box'][0]+3<p[0]<q['box'][2]-3 and q['box'][1]<p[1]<q['box'][3]-8 for q in others)
        if not covered:accepted.append(j)
    if not accepted:return None,'support_possibly_occluded',selected
    p=kp[accepted,:2].mean(axis=0)
    if not inside(stage(p)):return None,'support_off_mat',accepted
    return p,kind,accepted

def evaluate_all():
    base=read(TRACK_DIR/'tracks.json')['roster'];labels=read(LABEL_DIR/'labels.json');cache=read(OUT/'wholebody.json');assert len(cache)==300
    assoc=[{int(i):o for i,o in cache[str(k)].items()} for k in range(300)]
    cfgs=[dict(mode=m,confidence=c) for m in ['feet_midpoint','feet_support','feet_knees'] for c in [.3,.5]]
    results=[];runs=[]
    for cfg in cfgs:
        data=maps(base,assoc,cfg,anchor_fn=contact);assert frozen_hash(data)==frozen_hash(base)
        results.append({'settings':cfg,'raw_validation':raw_metrics(data,labels,VAL),'map_validation':evaluate(data,labels,VAL)});runs.append(data)
    index=max(range(len(results)),key=lambda i:(results[i]['raw_validation']['joint_f1'],results[i]['raw_validation']['correct_within_3pct']))
    chosen=runs[index]
    write('selection.json',{'selected':index,'rule':'Select by raw floor-point joint F1 on existing four validation frames. Six candidates: feet midpoint, raised-foot rejection, raised-foot rejection plus kneeling knees; confidence .3/.5. This is a second pretrained-model experiment after direct ankle projection failed. No final-section selection.','candidates':results})
    write('settings.json',cfgs[index]);write('tracks.json',{'current':base,'ankles':chosen})
    raw_base=copy.deepcopy(base)
    for row in raw_base['frames']:
        for o in row:
            p=ankle.bottom(np.array(o['box']));o['anchor_pixel']=p if o['status']=='observed' and inside(stage(p)) else None;o['anchor_reason']='box_bottom'
    scores={split:{name:{'raw':raw_metrics(d,labels,frames),'map':evaluate(d,labels,frames)} for name,d in [('current',raw_base),('feet',chosen)]} for split,frames in [('validation',VAL),('development_final',TEST)]}
    write('accuracy.json',scores);write('coverage.json',{n:dict(Counter(o['map_status'] for row in d['frames'] for o in row)) for n,d in [('current',base),('feet',chosen)]})
    write('motion.json',motion_stats(base,chosen));write('stationary.json',stationary(base,chosen,labels))
    # Keep all candidate outcomes available, including the knee proposal, without selecting on them.
    write('all_final_diagnostics.json',[{'settings':c,'raw':raw_metrics(d,labels,TEST),'map':evaluate(d,labels,TEST)} for c,d in zip(cfgs,runs)])
    controls=[maps(base,assoc,dict(mode='ankle_support',confidence=c)) for c in [.3,.5]]
    control_val=[raw_metrics(d,labels,VAL) for d in controls]
    ci=max(range(2),key=lambda i:control_val[i]['joint_f1']);control=controls[ci]
    write('rtmw_ankle_control.json',{'scope':'Same RTMW model and frozen boxes; use ankle indices15/16 instead of toe/heel indices17..22. Threshold selected on validation.','confidence':[.3,.5][ci],
      'validation':{'raw':control_val[ci],'map':evaluate(control,labels,VAL)},'development_final':{'raw':raw_metrics(control,labels,TEST),'map':evaluate(control,labels,TEST)}})
    write('provenance.json',{'same_detector_boxes_ids_status':frozen_hash(base)==frozen_hash(chosen),'frozen_sha256':frozen_hash(base),'model_url':URL,'no_retraining':True,'map_smoothing':'Unchanged median and bounded EMA. No Kalman. No box-bottom fallback.','limits':['Toes/heels/knees may be predicted through occlusion; confidence is not contact probability.','Top-down pose on overlapping boxes can follow the wrong body.','Kneeling support and two-foot support are different targets.','Previously reviewed development clip with approximate references.']})
    print(json.dumps({'selected':cfgs[index],'final_raw':scores['development_final']['feet']['raw'],'final_map':scores['development_final']['feet']['map'],'validation':results[index]['raw_validation']},default=lambda x:x.item() if isinstance(x,np.generic) else x,indent=2),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='infer':infer()
    else:evaluate_all()
