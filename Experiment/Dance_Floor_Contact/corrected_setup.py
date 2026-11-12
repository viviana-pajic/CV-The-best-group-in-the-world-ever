"""Prepare an independently reviewed short-clip fine-tuning pilot."""
import json
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from infer import WEIGHTS
from compare import overlap

BASE=Path(__file__).resolve().parent.parent
OUT=BASE/'Dance_Corrected_Pilot'
SOURCE=BASE/'Dance_Controlled_Test'
TRAIN=list(range(0,144,6));VAL=[165,175,185,195];TEST=[225,235,245,255,265,275,285,299]
FRAMES=TRAIN+VAL+TEST
TILES=[(0,280,610,610),(335,280,945,610),(670,280,1280,610)]
def read(p):return json.loads(p.read_text())
def get_image(k):
    im=cv2.imread(str(OUT/'images'/f'{k:03}.png'))
    assert im is not None,k
    return im
def prepare():
    (OUT/'images').mkdir(parents=True,exist_ok=True);(OUT/'audit').mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(SOURCE/'source.mp4'))
    for k in range(300):
        ok,im=cap.read();assert ok
        if k in FRAMES:cv2.imwrite(str(OUT/'images'/f'{k:03}.png'),im)
    cap.release()
    manifest=[{'frame':k,'source_frame':4420+k,'split':'train' if k in TRAIN else 'val' if k in VAL else 'test'} for k in FRAMES]
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    protocol={'frames':{'train':TRAIN,'val':VAL,'test':TEST},'training_scope':'Only this clip, pretrained COCO initialization. No labels from previous learned models used as accepted ground truth.',
      'selection':'Fixed before corrected labels and training. First5.52s train,6.6–7.8s validation,last3s test; temporal gaps separate splits.',
      'labels':'Review every dancer on each selected keyframe. Visible-extent box; support point between grounded feet or knees when discernible. Hidden/uncertain support has no coordinate and visibility0.',
      'not_labeled':'The intervening264 frames are inference/replay frames, not fabricated training labels.',
      'evaluation':'One-to-one box matching at IoU0.5; visible-contact map error and fraction correctly localized across ALL visible references; predictions on hidden supports reported separately. Checkpoint and confidence selection use validation only.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    model=YOLO(WEIGHTS/'yolo26m.pt');props={}
    for k in FRAMES:
        im=get_image(k);candidates=[]
        for (l,t,r,b),result in zip(TILES,model.predict([im[t:b,l:r] for l,t,r,b in TILES],imgsz=960,conf=.06,classes=[0],device=0,verbose=False)):
            for row in result.boxes.data.cpu().numpy():
                a=row.astype(float);a[:4]+=[l,t,l,t]
                if (l>0 and a[0]<l+5) or (r<1280 and a[2]>r-5):continue
                if any(overlap(a,q)>.65 for q in candidates):continue
                candidates.append(a)
        props[str(k)]=[a.tolist() for a in candidates]
    (OUT/'proposals.json').write_text(json.dumps(props))
    # Fixed floor neighbourhoods are association proposals only, reviewed below.
    anchors=np.array([[230,449],[251,477],[267,536],[362,425],[408,447],[421,488],[484,425],[510,534],[546,448],[631,481],[638,427],[723,452],[730,532],[752,415],[823,488],[856,425],[867,458],[1048,552],[1096,455],[1101,491]])
    sam=read(SOURCE/'sam.json');draft={}
    for k in FRAMES:
        boxes=np.array(props[str(k)])
        cost=np.zeros((20,len(boxes)))
        for i,p in enumerate(anchors):
            for j,b in enumerate(boxes):
                q=np.array([(b[0]+b[2])/2,b[3]])
                cost[i,j]=np.linalg.norm((q-p)*[.7,1.4])+max(b[0]-p[0],p[0]-b[2],0)*.5
        ii,jj=linear_sum_assignment(cost);links=dict(zip(ii,jj));items=[]
        for i,p in enumerate(anchors):
            candidate=next((o['box'] for o in sam[k] if o['id']==i+1),[p[0]-20,p[1]-60,p[0]+20,p[1]])
            if i in links and cost[i,links[i]]<70:candidate=boxes[links[i],:4]
            items.append({'id':i+1,'box':list(map(lambda v:round(float(v)),candidate)),'floor':None,'contact':'unreviewed','reviewed':False})
        draft[str(k)]=items
    (OUT/'draft.json').write_text(json.dumps(draft,indent=2))
    print('Prepared',len(FRAMES),'review frames')
def sheets(frames=None,raw=False):
    labels=read(OUT/('labels.json' if (OUT/'labels.json').exists() else 'draft.json'))
    for k in frames or FRAMES:
        im=get_image(k)
        for x in range(150,1201,50):
            cv2.line(im,(x,280),(x,600),(95,95,95),1);cv2.putText(im,str(x),(x+2,294),0,.30,(255,255,255),1)
        for y in range(300,601,25):
            cv2.line(im,(150,y),(1200,y),(80,80,80),1);cv2.putText(im,str(y),(151,y-2),0,.30,(255,255,255),1)
        if not raw:
            for o in labels[str(k)]:
                if o.get('outside'):continue
                x,y,x2,y2=o['box'];c=(80,220,110) if o['reviewed'] else (50,180,255)
                cv2.rectangle(im,(x,y),(x2,y2),c,1)
                cv2.rectangle(im,(x,y-14),(x+22,y),(20,20,20),-1)
                cv2.putText(im,str(o['id']),(x+1,y-3),0,.38,c,1)
                if o['floor']:cv2.drawMarker(im,tuple(o['floor']),(255,255,0),cv2.MARKER_CROSS,9,1)
        im=im[280:605,145:1205]
        im=cv2.resize(im,(1590,488));cv2.putText(im,f'Frame {k} / +{k/25:.2f}s',(9,477),0,.6,(255,255,255),2)
        cv2.imwrite(str(OUT/'audit'/f'{k:03}{"_raw" if raw else ""}.jpg'),im)
if __name__=='__main__':
    import sys
    if len(sys.argv)<2 or sys.argv[1]=='prepare':prepare()
    else:sheets([int(x) for x in sys.argv[2:]] or None,sys.argv[1]=='raw')
