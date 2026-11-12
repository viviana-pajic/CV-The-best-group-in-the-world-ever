"""Exploratory body-based floor estimates for occluded support, never measurements."""
import sys,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
from compare import match_boxes,stage
from evaluate_support import read,write
from ankle_pilot import angle

def landmark(pose,kind):
    kp=np.array(pose['kp']);ii={'nose':[0],'shoulders':[5,6],'hips':[11,12]}[kind]
    if min(kp[ii,2])<.6:return None
    return kp[ii,:2].mean(0)

def upright(pose):
    kp=np.array(pose['kp'])
    for h,k,a in [(11,13,15),(12,14,16)]:
        if min(kp[[h,k,a],2])>.3 and angle(kp[h,:2],kp[k,:2],kp[a,:2])>150 and kp[k,1]>kp[h,1]+10:return True
    return False

def transform(H,p):
    q=H@np.r_[p,1.];return q[:2]/q[2]

def main():
    cv2.setRNGSeed(17);folder=OUT/'head_plane';folder.mkdir(exist_ok=True);dataset=read(OUT/'data/development_v1.json')['controlled'];poses=read(ROOT/'Dance_Ankle_Pilot/pose_candidates.json');labels=dataset['labels'];examples={}
    for split in ['train','val']:
        rows=[]
        for k in dataset[split]:
            gt=[o for o in labels[str(k)] if o['box'] is not None];pp=poses[str(k)];links=match_boxes([o['box'] for o in gt],[o['box'] for o in pp],.5)
            for i,j in links.items():rows.append({'frame':k,'label':gt[i],'pose':pp[j]})
        examples[split]=rows
    results=[]
    for kind in ['nose','shoulders','hips']:
        training=[r for r in examples['train'] if r['label']['support_type']=='feet' and landmark(r['pose'],kind) is not None and upright(r['pose'])];a=np.array([landmark(r['pose'],kind) for r in training],np.float32);b=np.array([r['label']['floor'] for r in training],np.float32)
        if len(a)<8:continue
        for method in ['affine','homography']:
            if method=='homography':H,inliers=cv2.findHomography(a,b,cv2.RANSAC,8)
            else:
                M,inliers=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=8);H=np.vstack([M,[0,0,1]]) if M is not None else None
            if H is None:continue
            records=[]
            for r in examples['val']:
                o=r['label'];p=landmark(r['pose'],kind)
                if p is None or not upright(r['pose']):continue
                q=transform(H,p);v=stage(q)
                if not (0<=v[0]<=100 and 0<=v[1]<=62.5):continue
                records.append({'frame':r['frame'],'id':o['id'],'support_type':o['support_type'],'contact':o['contact'],'point':q.tolist(),'error':float(np.linalg.norm(v-stage(o['floor']))) if o['floor'] is not None else None})
            errors=[r['error'] for r in records if r['error'] is not None];correct=sum(e<=3 for e in errors);row={'kind':kind,'method':method,'H':H.tolist(),'training_matches':len(a),'ransac_inliers':int(inliers.sum()),'validation_available_visible':len(errors),'correct_3pct':correct,'median_error':float(np.median(errors)) if errors else None,'records':records};results.append(row);print(kind,method,'correct',correct,'of',len(errors),'median',row['median_error'],flush=True)
    write(folder/'results.json',{'scope':'Exploratory standing-height model trained only on controlled training feet labels, using cached generic COCO poses. Outputs are body-based estimates even when feet are hidden. A roughly constant upright body height is assumed; this is not a contact detector. Validation uses the same development clip, with failures/missing cases retained.','variants':results})

if __name__=='__main__':main()
