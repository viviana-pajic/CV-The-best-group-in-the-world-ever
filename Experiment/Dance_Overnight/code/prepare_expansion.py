"""Blank manual contact-review sheets; old model points are discarded."""
import json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight/data/expansion';OUT.mkdir(parents=True,exist_ok=True)
TRAIN=[3290,3332,3360,3465,3507,3535,3815,3857,3885,4115,4157,4185,4315,4357,4385,5315,5357,5385]
VAL=[3640,3682,3710]

def main(proposals=None):
    old=proposals if proposals is not None else json.loads((ROOT/'Dance_Training_Pilot/pilot_labels.json').read_text());blank={}
    for folder in ['images','review','crops']:(OUT/folder).mkdir(exist_ok=True)
    for k in TRAIN+VAL:
        im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{k:06}.png'));assert im is not None
        cv2.imwrite(str(OUT/'images'/f'{k}.png'),im)
        blank[str(k)]=[{'id':i+1,'box':o['box'],'floor':None,'contact':'unreviewed','support_type':None,'outside':False,'reviewed':False,'identity_scope':'image-local proposal index; not cross-frame identity'} for i,o in enumerate(old[f'frame_{k:06}.png'])]
        review=im.copy()
        for o in blank[str(k)]:
            x1,y1,x2,y2=map(round,o['box']);cv2.rectangle(review,(x1,y1),(x2,y2),(100,210,100),1)
            cv2.rectangle(review,(x1,y1-18),(x1+28,y1),(0,0,0),-1);cv2.putText(review,str(o['id']),(x1+2,y1-3),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.imwrite(str(OUT/'review'/f'{k}.jpg'),review)
        n=len(blank[str(k)]);board=np.full((((n+3)//4)*300,4*380,3),25,np.uint8)
        for j,o in enumerate(blank[str(k)]):
            bx=(j%4)*380;by=(j//4)*300;x1,y1,x2,y2=o['box'];l=int(np.clip((x1+x2)/2-50,0,1180));t=int(np.clip(y2-58,270,620));r=l+100;b=t+80
            patch=cv2.resize(im[t:b,l:r],(300,240));ox=bx+55;oy=by+40
            board[oy:oy+240,ox:ox+300]=patch
            cv2.putText(board,f"{o['id']}  /  source {k}",(bx+8,by+21),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1)
            for x in range((l//20+1)*20,r,20):
                xx=ox+(x-l)*3;cv2.line(board,(xx,oy),(xx,oy+240),(85,85,85),1);cv2.putText(board,str(x),(xx-12,oy-5),cv2.FONT_HERSHEY_SIMPLEX,.32,(240,240,240),1)
            for y in range((t//10+1)*10,b,10):
                yy=oy+(y-t)*3;cv2.line(board,(ox,yy),(ox+300,yy),(85,85,85),1);cv2.putText(board,str(y),(bx+16,yy+4),cv2.FONT_HERSHEY_SIMPLEX,.34,(240,240,240),1)
            # Highlight the target person's original box inside this local context.
            a=(int(ox+(x1-l)*3),int(oy+(y1-t)*3));c=(int(ox+(x2-l)*3),int(oy+(y2-t)*3))
            a=(max(ox,a[0]),max(oy,a[1]));c=(min(ox+299,c[0]),min(oy+239,c[1]));cv2.rectangle(board,a,c,(80,180,80),1)
        cv2.imwrite(str(OUT/'crops'/f'{k}.jpg'),board)
    (OUT/'blank.json').write_text(json.dumps(blank,indent=2));(OUT/'manifest.json').write_text(json.dumps({'train':TRAIN,'val':VAL,'holdout_excluded':[4800,5080],'status':'BOX PROPOSALS ONLY. Every box/contact requires visual review; previous model-derived support points discarded.'},indent=2))
    print('Prepared',len(blank),'frames and',sum(map(len,blank.values())),'blank person proposals.',flush=True)

if __name__=='__main__':main()
