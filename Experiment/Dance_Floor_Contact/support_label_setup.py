"""Prepare blank contact-review boards. Existing coordinates are NOT inherited."""
import json,copy
from pathlib import Path
import cv2,numpy as np
from corrected_setup import BASE,OUT as OLD,SOURCE,read

OUT=BASE/'Dance_Support_Training_Pilot';OUT.mkdir(exist_ok=True)
TRAIN=list(range(0,144,12));VAL=[165,175,185,195];TEST=[225,245,265,285,299]
FRAMES=TRAIN+VAL+TEST

def prepare():
    labels=read(OLD/'labels.json');manifest=[];blank={}
    for folder in ['images','review','person_review']:(OUT/folder).mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(SOURCE/'source.mp4'))
    for k in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES,k);ok,im=cap.read();assert ok
        cv2.imwrite(str(OUT/'images'/f'{k:03}.png'),im)
        items=[]
        for o in labels[str(k)]:
            items.append({'id':o['id'],'box':o['box'],'outside':bool(o.get('outside')),'floor':None,'contact':'unreviewed','support_type':None,'reviewed':False})
        blank[str(k)]=items
        split='train' if k in TRAIN else 'val' if k in VAL else 'test';manifest.append({'frame':k,'split':split})
        crop=im[300:600,140:1180].copy();scale=2
        for x in range(150,1180,25):
            xx=x-140;cv2.line(crop,(xx,0),(xx,300),(90,90,90),1);cv2.putText(crop,str(x),(xx+1,12),cv2.FONT_HERSHEY_SIMPLEX,.27,(255,255,255),1)
        for y in range(325,600,25):
            yy=y-300;cv2.line(crop,(0,yy),(1040,yy),(90,90,90),1);cv2.putText(crop,str(y),(1,yy-2),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
        for o in items:
            if o['box'] is None:continue
            x1,y1,x2,y2=map(round,o['box']);x1-=140;x2-=140;y1-=300;y2-=300
            cv2.rectangle(crop,(x1,y1),(x2,y2),(80,190,80),1)
            cv2.rectangle(crop,(x1,y1-14),(x1+23,y1),(0,0,0),-1);cv2.putText(crop,str(o['id']),(x1+1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
        cv2.imwrite(str(OUT/'review'/f'{k:03}.jpg'),cv2.resize(crop,(2080,600)))
        # Each numbered crop keeps global coordinate ticks for precise point placement.
        board=np.full((5*220,4*330,3),25,np.uint8)
        for j,o in enumerate(items):
            bx=(j%4)*330;by=(j//4)*220
            cv2.putText(board,f"D{o['id']:02} / frame {k}",(bx+8,by+18),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1)
            if o['box'] is None:continue
            x1,y1,x2,y2=o['box'];l=max(0,int((x1+x2)/2-75));r=min(1280,l+150);t=max(300,int(y2-75));b=min(650,t+95)
            patch=im[t:b,l:r].copy()
            for x in range((l//20+1)*20,r,20):
                cv2.line(patch,(x-l,0),(x-l,b-t),(90,90,90),1);cv2.putText(patch,str(x),(x-l,9),cv2.FONT_HERSHEY_SIMPLEX,.23,(255,255,255),1)
            for y in range((t//20+1)*20,b,20):
                cv2.line(patch,(0,y-t),(r-l,y-t),(90,90,90),1);cv2.putText(patch,str(y),(1,y-t-1),cv2.FONT_HERSHEY_SIMPLEX,.23,(255,255,255),1)
            board[by+25:by+215,bx+10:bx+310]=cv2.resize(patch,(300,190))
        cv2.imwrite(str(OUT/'person_review'/f'{k:03}.jpg'),board)
    cap.release()
    (OUT/'blank.json').write_text(json.dumps(blank,indent=2));(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'protocol.json').write_text(json.dumps({'train':TRAIN,'validation':VAL,'test':TEST,'label_definition':'Stance anchor on the mat: center between visibly planted feet when standing; planted foot if the other foot is raised; center between visibly supporting knees when kneeling. Do not use ankle joints or box bottom as labels. Hands are not included in the knee/foot stance anchor. If the supporting part or contact is not clear, leave coordinate absent and mark hidden/uncertain.','point_labels':'Fresh visual review on every selected frame; no inherited or interpolated coordinates. Boxes reused from prior reviewed annotations. Labels are assistant-reviewed, not independent human ground truth.','training':'Train a separate per-person floor-support and reportability estimator. Keep detector, IDs and calibration fixed. Temporal gap between splits; already reviewed clip is development, not a fresh generalization benchmark.'},indent=2))
    print('Prepared',len(FRAMES),'blank review frames',flush=True)

if __name__=='__main__':prepare()
