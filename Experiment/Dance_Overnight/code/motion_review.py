"""Review sheets for dense development-only front-row support trajectories."""
import cv2,numpy as np,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight/data/motion_review';OUT.mkdir(exist_ok=True)
IDS=[3,8,13,18];CENTERS=[250,505,745,1060]
GROUPS=[[167,169,171,173],[177,179,181,183],[187,189,191,193]]
for group in GROUPS:
    board=np.full((len(group)*300,1520,3),25,np.uint8)
    for row,k in enumerate(group):
        im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{4420+k:06}.png'))
        for j,(ident,center) in enumerate(zip(IDS,CENTERS)):
            bx=j*380;by=row*300;l=center-50;t=490;ox=bx+55;oy=by+40
            board[oy:oy+240,ox:ox+300]=cv2.resize(im[t:t+80,l:l+100],(300,240))
            cv2.putText(board,f'frame {k} / ID {ident}',(bx+8,by+21),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1)
            for x in range((l//10+1)*10,l+100,10):
                xx=ox+(x-l)*3;cv2.line(board,(xx,oy),(xx,oy+240),(80,80,80),1);cv2.putText(board,str(x),(xx-10,oy-5),cv2.FONT_HERSHEY_SIMPLEX,.3,(240,240,240),1)
            for y in range(500,570,10):
                yy=oy+(y-t)*3;cv2.line(board,(ox,yy),(ox+300,yy),(80,80,80),1);cv2.putText(board,str(y),(bx+15,yy+4),cv2.FONT_HERSHEY_SIMPLEX,.34,(240,240,240),1)
    cv2.imwrite(str(OUT/f'{group[0]}.jpg'),board)
print('Three motion review sheets prepared')
