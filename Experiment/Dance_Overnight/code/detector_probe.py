"""Compare detector resolution, merge threshold and confidence on development."""
import sys,argparse,time,json
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight/detector';sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
from corrected_setup import TILES
from compare import overlap,match_boxes
from evaluate_support import read,write
VAL=[165,175,185,195,285,299]

def candidates(model,im,imgsz=960,device='cuda'):
    result=[]
    for (l,t,r,b),pred in zip(TILES,model.predict([im[t:b,l:r] for l,t,r,b in TILES],imgsz=imgsz,conf=.06,classes=[0],device=device,verbose=False)):
        for row in pred.boxes.data.cpu().numpy():
            a=row.astype(float);a[:4]+=[l,t,l,t]
            if (l>0 and a[0]<l+5) or (r<1280 and a[2]>r-5):continue
            result.append({'box':a[:4].tolist(),'score':float(a[4])})
    return sorted(result,key=lambda o:o['score'],reverse=True)

def merge(rows,threshold=.65):
    keep=[]
    for o in sorted(rows,key=lambda o:o['score'],reverse=True):
        if not any(overlap(np.array(o['box']),np.array(p['box']))>threshold for p in keep):keep.append(o)
    return keep

def score(pred,labels,threshold):
    total=matched=precise=count=0;perframe=[]
    for k in VAL:
        gt=[o['box'] for o in labels[str(k)] if not o.get('outside') and o['box'] is not None];pp=[o['box'] for o in pred[str(k)] if o['score']>=threshold];n=len(match_boxes(gt,pp,.5));m=len(match_boxes(gt,pp,.75));total+=len(gt);matched+=n;precise+=m;count+=len(pp);perframe.append({'frame':k,'refs':len(gt),'predictions':len(pp),'matched50':n,'matched75':m})
    return {'references':total,'predictions':count,'matched50':matched,'matched75':precise,'precision50':matched/max(count,1),'recall50':matched/max(total,1),'f1':2*matched/max(total+count,1),'per_frame':perframe}

def probe(name,device='cpu'):
    weights=ROOT/'Dance_Corrected_Pilot/runs/detect/weights/best.pt' if name=='previous' else OUT/'runs'/name/'weights/best.pt'
    assert weights.exists(),weights
    if name!='previous':assert (weights.parents[1]/'complete.json').exists(),'Wait for completed training'
    out=OUT/'probe';out.mkdir(exist_ok=True);labels=read(ROOT/'Dance_Support_Training_Pilot/labels.json');model=YOLO(weights);summary=[]
    for size in [960,1280]:
        path=out/f'{name}_{size}_raw.json';start=time.perf_counter()
        if path.exists():raw=read(path)
        else:
            raw={}
            cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'))
            for k in VAL:
                cap.set(cv2.CAP_PROP_POS_FRAMES,k);ok,im=cap.read();assert ok;raw[str(k)]=candidates(model,im,size,device)
            cap.release();write(path,raw)
        for nms in [.55,.65,.75,.85]:
            pred={k:merge(row,nms) for k,row in raw.items()}
            for confidence in [.1,.15,.25,.35,.45,.55,.65]:summary.append({'model':name,'weights':str(weights),'imgsz':size,'merge_iou':nms,'confidence':confidence,'validation':score(pred,labels,confidence)})
        print(name,size,'done',round(time.perf_counter()-start,1),flush=True)
    summary.sort(key=lambda r:(r['validation']['f1'],r['validation']['matched75']),reverse=True);write(out/f'{name}_results.json',summary);print('BEST',summary[0],flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);cv2.setNumThreads(2);p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--device',default='cpu');a=p.parse_args();probe(a.model,a.device)
