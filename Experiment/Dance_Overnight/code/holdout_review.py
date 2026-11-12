"""Source-only magnifications for the reserved annotation audit."""
from pathlib import Path
import cv2,json
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'Dance_Overnight/data/holdout'
for k in [4975,5010,5045,5080]:
    im=cv2.imread(str(D/'images'/f'{k}.png'))
    for name,l,r in [('left',100,500),('middle',480,820),('right',800,1170)]:
        top=300;bottom=580;scale=3
        out=cv2.resize(im[top:bottom,l:r],((r-l)*scale,(bottom-top)*scale))
        for x in range((l//25+1)*25,r,25):
            xx=(x-l)*scale;cv2.line(out,(xx,0),(xx,out.shape[0]),(100,100,100),1);cv2.putText(out,str(x),(xx+2,15),0,.4,(255,255,255),1)
        for y in range(top,bottom,25):
            yy=(y-top)*scale;cv2.line(out,(0,yy),(out.shape[1],yy),(100,100,100),1);cv2.putText(out,str(y),(2,yy+14),0,.4,(255,255,255),1)
        cv2.imwrite(str(D/f'{k}_{name}.jpg'),out)
print('Source-only review zooms written')
