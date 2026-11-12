"""Add reviewed kneeling boxes to the detector's development training set."""
import sys,json,time,hashlib,argparse
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight/detector';sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
from corrected_setup import TILES
from infer import WEIGHTS
from evaluate_support import read,write
TRAIN=list(range(0,144,6))+[225,245];VAL=[165,175,185,195,285,299]

def export():
    labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');fresh=read(ROOT/'Dance_Support_Training_Pilot/labels.json');labels['225']=fresh['225'];labels['245']=fresh['245'];root=OUT/'dataset'
    for split,frames in [('train',TRAIN),('val',VAL)]:
        for folder in ['images','labels']:(root/folder/split).mkdir(parents=True,exist_ok=True)
        for k in frames:
            im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{4420+k:06}.png'));assert im is not None
            for tile,(l,t,r,b) in enumerate(TILES):
                w,h=r-l,b-t;lines=[]
                for o in labels[str(k)]:
                    if o.get('outside') or o['box'] is None:continue
                    a,c,d,e=o['box'];a2,c2,d2,e2=max(a,l),max(c,t),min(d,r),min(e,b)
                    if d2-a2<6 or e2-c2<8 or (d2-a2)*(e2-c2)<.1*(d-a)*(e-c):continue
                    xywh=[(a2+d2-2*l)/(2*w),(c2+e2-2*t)/(2*h),(d2-a2)/w,(e2-c2)/h];lines.append('0 '+' '.join(f'{v:.7f}' for v in xywh))
                # Repeat the two newly labeled floor-work frames to reduce posture imbalance.
                repeats=4 if split=='train' and k in [225,245] else 1
                for rep in range(repeats):
                    name=f'{k:03}_tile{tile}_r{rep}';cv2.imwrite(str(root/'images'/split/(name+'.jpg')),im[t:b,l:r],[cv2.IMWRITE_JPEG_QUALITY,98]);(root/'labels'/split/(name+'.txt')).write_text('\n'.join(lines)+'\n')
    (root/'data.yaml').write_text(f"path: '{root.as_posix()}'\ntrain: images/train\nval: images/val\nnames: [dancer]\n")
    write(OUT/'dataset_provenance.json',{'train_frames':TRAIN,'validation_frames':VAL,'source_first':4420,'unique_train_frames':len(TRAIN),'effective_train_tiles':96,'new_floor_work_frames':[225,245],'oversampling':'Four copies per kneeling frame, randomized augmentations during training. Copies are not additional independent annotations.','holdout_excluded':[4800,5080],'boxes':'Prior reviewed visible-extent boxes, plus latest support-pilot reviewed225/245 boxes. Fully hidden people have no invented box. Expansion image-local proposals are not used as exhaustive detector labels.','labels_sha256':hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest()})
    print('Exported',len(TRAIN),'unique training frames and',len(VAL),'development validation frames',flush=True)

def train(name,freeze,lr,epochs=60):
    out=OUT/'runs'/name
    if (out/'complete.json').exists():return
    torch.set_num_threads(4);start=time.perf_counter();model=YOLO(WEIGHTS/'yolo26m.pt');source=model.model.model[-1]
    def keep_person(trainer):
        with torch.no_grad():
            for net in (trainer.model,trainer.ema.ema):
                for attr in ('cv3','one2one_cv3'):
                    for src,dst in zip(getattr(source,attr),getattr(net.model[-1],attr)):
                        dst[-1].weight.copy_(src[-1].weight[:1]);dst[-1].bias.copy_(src[-1].bias[:1])
    model.add_callback('on_pretrain_routine_end',keep_person)
    result=model.train(data=str(OUT/'dataset/data.yaml'),imgsz=960,epochs=epochs,batch=2,workers=0,device=0,amp=True,seed=26,deterministic=True,freeze=freeze,optimizer='AdamW',lr0=lr,nbs=16,warmup_epochs=2,patience=15,mosaic=0,mixup=0,copy_paste=0,scale=.10,translate=.025,hsv_h=.005,hsv_s=.15,hsv_v=.15,fliplr=.5,plots=True,project=str(OUT/'runs'),name=name,exist_ok=True,verbose=False)
    write(out/'complete.json',{'name':name,'freeze':freeze,'lr':lr,'epochs_requested':epochs,'seconds':time.perf_counter()-start,'weights':str(model.trainer.best),'validation':result.results_dict,'initialization':'Generic COCO YOLO26m; person classifier channel preserved. No older choreography checkpoint used.'});print('FINISHED detector',name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['export','frozen','full','both']);args=p.parse_args()
    if args.mode=='export':export()
    else:
        if args.mode in ['frozen','both']:train('posture_frozen',10,.0004)
        if args.mode in ['full','both']:train('posture_full',0,.00012)
