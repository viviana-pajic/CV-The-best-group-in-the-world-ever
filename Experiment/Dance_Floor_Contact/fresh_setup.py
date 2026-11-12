"""Prepare a frozen fresh test and train a separate contact model on dancer crops."""
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from infer import PILOT, ROOT, read, image

OUT=ROOT.parent/'Dance_Fresh_Pilot'
CLIPS={'overlap':range(3925,4051),'floor':range(5200,5301)}

def crop_bounds(box):
    x1,y1,x2,y2=box[:4]; w=x2-x1; h=y2-y1
    return [max(0,int(x1-.10*w)),max(0,int(y1-.07*h)),min(1280,int(x2+.10*w)),min(720,int(y2+.10*h))]

def prepare():
    OUT.mkdir(exist_ok=True)
    manifest=read(PILOT/'split_manifest.json')
    assert not ({f for c in CLIPS.values() for f in c}&{r['source_frame'] for r in manifest})
    (OUT/'protocol.json').write_text(json.dumps({'clips':{k:[c.start,c.stop-1] for k,c in CLIPS.items()},
        'comparison':'Frozen hybrid and smoothed boxes; separately trained crop contact model',
        'test_usage':'Fresh sections excluded from training and checkpoint selection',
        'manual_frames':[3950,4025,5250,5290]},indent=2))
    labels=read(PILOT/'pilot_labels.json')
    for r in manifest:
        if r['split'] not in ('train','val'):continue
        im=image(r['source_frame'])
        for i,o in enumerate(labels[r['filename']]):
            if not o['floor']:continue
            l,t,rr,b=crop_bounds(o['box']); w,h=rr-l,b-t
            x,y=o['floor']
            if not(l<=x<rr and t<=y<b):continue
            base=OUT/'crops'; split=r['split']; name=f"{r['source_frame']}_{i}"
            for folder in ('images','labels'):(base/folder/split).mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(base/'images'/split/(name+'.jpg')),im[t:b,l:rr])
            a,c,d,e=o['box']
            values=[(a+d-2*l)/(2*w),(c+e-2*t)/(2*h),(d-a)/w,(e-c)/h,(x-l)/w,(y-t)/h,2]
            (base/'labels'/split/(name+'.txt')).write_text('0 '+' '.join(map(str,values))+'\n')
    base=OUT/'crops'
    (base/'data.yaml').write_text(f"path: '{base.as_posix()}'\ntrain: images/train\nval: images/val\nnames: [dancer]\nkpt_shape: [1, 3]\nflip_idx: [0]\n")

def train():
    model=YOLO(PILOT/read(PILOT/'run_pose.json')['weight'])
    result=model.train(data=str(OUT/'crops/data.yaml'),imgsz=320,epochs=12,batch=16,device=0,workers=0,
        freeze=10,optimizer='AdamW',lr0=.0002,nbs=16,warmup_epochs=1,patience=5,
        mosaic=0,mixup=0,scale=.1,translate=.03,fliplr=.5,seed=26,
        project=str(OUT/'runs'),name='crop_contact',plots=True)
    (OUT/'training.json').write_text(json.dumps({'weights':str(model.trainer.best),
        'validation':result.results_dict,'note':'Existing single-point network adapted to person crops; assisted labels.'},indent=2))

if __name__=='__main__':
    prepare();train()
