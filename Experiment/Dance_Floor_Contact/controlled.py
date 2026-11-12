"""Twelve-second, first-frame-only initialized YOLO/SAM 2 comparison."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import SAM2VideoPredictor
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from infer import image,WEIGHTS
from compare import overlap,match_boxes,stage,smooth

OUT=Path(__file__).resolve().parent.parent/'Dance_Controlled_Test'
FIRST,COUNT=4420,300
# Visually placed torso prompts for all 20 people, including partially occluded ones.
MARKS=[
([211,394,246,452],[228,422]),([238,424,274,480],[252,449]),([245,468,298,540],[269,503]),
([346,382,382,429],[362,403]),([365,341,442,449],[407,365]),([336,336,491,493],[425,381]),
([469,387,501,429],[483,405]),([473,465,551,539],[512,503]),([519,328,570,453],[548,371]),
([559,344,692,488],[630,383]),([611,324,666,431],[641,343]),([659,333,785,460],[723,375]),
([686,474,780,539],[730,503]),([737,371,770,419],[752,397]),([757,352,899,495],[822,395]),
([842,377,870,429],[854,401]),([840,339,918,466],[867,369]),([1000,480,1101,560],[1047,523]),
([1076,402,1119,460],[1094,430]),([1081,437,1137,495],[1104,469])]

def prepare():
    OUT.mkdir(exist_ok=True)
    seeds=[{'id':i+1,'box':b,'point':p} for i,(b,p) in enumerate(MARKS)]
    (OUT/'seeds.json').write_text(json.dumps(seeds,indent=2))
    im=image(FIRST)
    for o in seeds:
        x,y=o['point'];cv2.rectangle(im,tuple(o['box'][:2]),tuple(o['box'][2:]),(255,180,70),1)
        cv2.circle(im,(x,y),3,(0,240,255),-1);cv2.putText(im,str(o['id']),(x+5,y),0,.4,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/'initialization.jpg'),im)
    writer=cv2.VideoWriter(str(OUT/'source.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(1280,720))
    for f in range(FIRST,FIRST+COUNT):writer.write(image(f))
    writer.release()
    (OUT/'protocol.json').write_text(json.dumps({'source_frames':[FIRST,FIRST+COUNT-1],'fps':25,
        'initial_people':20,'intervention':'Both methods initialized from the same 20 manually placed first-frame boxes; SAM also uses torso prompts. No later corrections.',
        'map':'Same smoothed box-bottom projection for both; SAM mask-bottom alternative also saved. Not a verified floor contact.'},indent=2))

def sam(limit=COUNT):
    seeds=json.loads((OUT/'seeds.json').read_text())
    centers=[o['point'] for o in seeds]
    prompts=[[p]+[q for j,q in enumerate(centers) if j!=i]+[[p[0],seeds[i]['box'][3]+12],[p[0],seeds[i]['box'][3]+30]] for i,p in enumerate(centers)]
    labels=[[1]+[0]*21 for _ in seeds]
    predictor=SAM2VideoPredictor(overrides=dict(model=str(OUT/'sam2.1_s.pt'),imgsz=1024,device=0,
        task='segment',mode='predict',conf=.1,quantize=16,verbose=False,save=False))
    rows=[];t=time.perf_counter();(OUT/'masks').mkdir(exist_ok=True)
    for k,result in enumerate(predictor(source=str(OUT/'source.mp4'),bboxes=[o['box'] for o in seeds],
                                     points=prompts,labels=labels,stream=True)):
        record=[];labelmap=np.zeros((720,1280),np.uint8)
        if result.masks is not None:
            for b,mask in zip(result.boxes.data.cpu().numpy(),result.masks.data.cpu().numpy()):
                ident=int(b[5])+1
                # Reject tiny detached speckles, retaining disconnected limbs.
                n,components,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
                if n<2:continue
                valid=np.where(stats[:,cv2.CC_STAT_AREA]>=16)[0]
                mask=np.isin(components,valid[valid!=0])
                ys,xs=np.where(mask)
                if len(xs)<24:continue
                floor_y=float(np.percentile(ys,99));footxs=xs[ys>=floor_y-3]
                record.append({'id':ident,'box':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],'raw_box':b[:4].tolist(),'area':len(xs),
                               'mask_bottom':[float(np.median(footxs)),floor_y]})
                labelmap[mask.astype(bool)]=ident
        cv2.imwrite(str(OUT/'masks'/f'{k:04}.png'),labelmap)
        rows.append(record)
        if k%25==0:print('SAM',k,'objects',len(record),'seconds',round(time.perf_counter()-t,1),flush=True)
        if k+1>=limit:break
    (OUT/('sam.json' if limit==COUNT else 'sam_smoke.json')).write_text(json.dumps(rows))
    (OUT/('sam_timing.json' if limit==COUNT else 'smoke_timing.json')).write_text(json.dumps({'seconds':time.perf_counter()-t,'frames':len(rows)}))
    print('SAM completed',len(rows),flush=True)

def yolo():
    started=time.perf_counter()
    model=YOLO(WEIGHTS/'yolo26m.pt')
    tracker=BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.10,new_track_thresh=.25,
                                       track_buffer=30,match_thresh=.8,fuse_score=True))
    seeds=json.loads((OUT/'seeds.json').read_text())
    init=np.array([o['box']+[.99,0] for o in seeds],np.float32)
    tracker.update(Boxes(init,(720,1280)))
    mapping={t.track_id:i+1 for i,t in enumerate(tracker.tracked_stracks)}
    rows=[[{'id':o['id'],'box':o['box'],'seed':True} for o in seeds]]
    windows=[(0,280,610,610),(335,280,945,610),(670,280,1280,610)]
    cap=cv2.VideoCapture(str(OUT/'source.mp4')); cap.read()
    for k in range(1,COUNT):
        ok,im=cap.read()
        assert ok, k
        candidates=[]
        for (l,t,r,b),pred in zip(windows,model.predict([im[t:b,l:r] for l,t,r,b in windows],imgsz=960,
                    classes=[0],conf=.10,device=0,quantize=16,verbose=False)):
            for row in pred.boxes.data.cpu().numpy():
                a=row.astype(float);a[:4]+=[l,t,l,t]
                if (l>0 and a[0]<l+5) or (r<1280 and a[2]>r-5):continue
                candidates.append(a)
        keep=[]
        for b in sorted(candidates,key=lambda b:b[4],reverse=True):
            if not any(overlap(b,q)>.65 for q in keep):keep.append(b)
        detected=np.asarray(keep,np.float32).reshape(-1,6)
        active=tracker.update(Boxes(detected,(720,1280)))
        record=[]
        for a in active:
            tid=int(a[4]); ident=mapping.setdefault(tid,100+tid)
            record.append({'id':ident,'box':a[:4].tolist(),'seed':False})
        rows.append(record)
        if k%25==0:print('YOLO',k,'tracks',len(record),flush=True)
    (OUT/'yolo.json').write_text(json.dumps(rows))
    cap.release()
    (OUT/'yolo_timing.json').write_text(json.dumps({'seconds':time.perf_counter()-started,'frames':len(rows)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','sam','yolo']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='smoke':sam(2)
    elif a.action=='sam':sam()
    else:yolo()
