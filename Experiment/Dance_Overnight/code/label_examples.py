"""Illustrate the annotation target using source crops, not generated imagery."""
from pathlib import Path
import json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';manifest=json.loads((OUT/'data/development_expanded.json').read_text());lookup={}
for d in manifest.values():
    for k in d['train']:
        for o in d['labels'][str(k)]:lookup[(d['source_first']+k,o['id'])]=o
for k,rows in json.loads((OUT/'data/holdout/labels.json').read_text()).items():
    for o in rows:lookup[(int(k),o['id'])]=o
examples=[('Planted feet',4540,3,'Sole planted foot, or midpoint of two'),('Supporting knees',3360,22,'Point at the supporting knees'),('Mixed knee + foot',4940,1,'Midpoint of grounded knee and foot'),('Seated contact',3535,13,'Point at the visible seat contact'),('Hands-only support',3521,13,'Point between supporting palms'),('Hidden contact',4516,2,'No point; contact = hidden'),('Airborne dancer',4235,102,'No point; contact = airborne'),('Unclear contact',3479,1,'No point; contact = uncertain')]
board=np.full((720,1600,3),(241,243,242),np.uint8)
for i,(title,frame,ident,caption) in enumerate(examples):
    o=lookup[(frame,ident)];im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{frame:06}.png'));x1,y1,x2,y2=map(round,o['box']);l=max(0,x1-20);r=min(1280,x2+20);t=max(0,y1-16);b=min(720,y2+22);patch=im[t:b,l:r].copy()
    if o['floor'] is not None:
        pt=(round(o['floor'][0]-l),round(o['floor'][1]-t));cv2.drawMarker(patch,pt,(40,255,90),cv2.MARKER_CROSS,11,2)
    scale=min(360/patch.shape[1],250/patch.shape[0]);patch=cv2.resize(patch,None,fx=scale,fy=scale);bx=(i%4)*400;by=(i//4)*360;px=bx+20+(360-patch.shape[1])//2;py=by+48+(250-patch.shape[0])//2;board[py:py+patch.shape[0],px:px+patch.shape[1]]=patch
    cv2.putText(board,title,(bx+20,by+28),0,.6,(45,65,60),1,cv2.LINE_AA);cv2.putText(board,caption,(bx+20,by+322),0,.43,(65,75,72),1,cv2.LINE_AA);cv2.putText(board,f'Source frame {frame} / instance {ident}',(bx+20,by+345),0,.38,(105,115,110),1,cv2.LINE_AA)
(OUT/'figures').mkdir(exist_ok=True);cv2.imwrite(str(OUT/'figures/label_examples.jpg'),board);print('Label examples written')
