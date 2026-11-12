"""Readable before/after video with honest measured/estimated map status."""
import sys,argparse,subprocess
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
from occlusion_deliver import source_frame,draw_map
from render_controlled import text,BG,INK,MUTED
from compare import bottom,pixel
from evaluate_support import read,write

def render(base,after,source,out,labels=None,pred=None,threshold=0,source_first=4420,title='SUPPORT POINTS / DEVELOPMENT COMPARISON',subtitle='Same detector and IDs | Offline smoothing uses nearby future frames',after_title='Learned support + offline smoothing',ring_description='Source rings: raw box-bottom / learned contact. Map dots: filtered positions. White crosses: visual reference.',before_title='Current / box-bottom map',before_pred=None,before_threshold=0):
    labels=labels or {};out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);raw=out.with_name(out.stem+'_raw.mp4');writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),25,(1600,1000));cap=cv2.VideoCapture(str(source));n=len(base['frames']);snapshots=sorted(set([0,n//4,n//2,3*n//4,n-1]));decoded=0
    for k in range(n):
        ok,im=cap.read();assert ok,(k,source);canvas=np.full((1000,1600,3),BG,np.uint8)
        text(canvas,title,24,36,.75,INK,2);text(canvas,f'Source {(source_first+k)/25:.2f}s | frame {k+1}/{n} | {subtitle}',24,66,.43,MUTED)
        for col,data in enumerate([base,after]):
            x=24+col*800;rows=data['frames'][k];text(canvas,before_title if col==0 else after_title,x,105,.56,INK,2)
            text(canvas,f"{sum(o['status']=='observed' for o in rows)} observed tracks | solid = model output, hollow = estimate",x,133,.41,MUTED)
            extra=data.get('diagnostics',[{}]*n)[k].get('unassigned_high_confidence_boxes',[]);annotated=source_frame(im,rows,extra)
            for o in rows:
                p=None
                if o['status']=='observed':
                    if col==0:
                        if before_pred is None:p=bottom(np.array(o['box']))
                        else:
                            pp=before_pred[k].get(str(o['id']),before_pred[k].get(o['id']))
                            if pp and pp['reportability']>=before_threshold:p=np.array(pp['point'])
                    elif pred:
                        pp=pred[k].get(str(o['id']),pred[k].get(o['id']))
                        if pp and pp['reportability']>=threshold:p=np.array(pp['point'])
                    elif o.get('raw_map') is not None:p=pixel(o['raw_map'])
                if p is not None:cv2.circle(annotated,tuple(np.rint(p).astype(int)),4,(80,235,135) if col else (0,170,255),2,cv2.LINE_AA)
            for g in labels.get(str(k),[]):
                if g.get('floor') is not None:cv2.drawMarker(annotated,tuple(g['floor']),(255,255,255),cv2.MARKER_CROSS,10,1)
            canvas[160:395,x:x+752]=cv2.resize(annotated[275:675],(752,235));text(canvas,'RELATIVE STAGE MAP | full black mat',x,447,.47,INK)
            cnt=draw_map(canvas,x+51,481,data['frames'],k,labels.get(str(k),[]));unknown=sum(o['map_status']=='unknown' for o in rows)
            text(canvas,f"{cnt['proxy']} solid | {cnt['estimated']} estimated | {unknown} unknown",x+51,934,.46,MUTED)
        text(canvas,ring_description,24,973,.44,MUTED)
        text(canvas,'Relative geometry, not surveyed meters. A hidden dancer or floor contact is not a measured position.',24,992,.40,MUTED)
        writer.write(canvas)
        if k in snapshots:cv2.imwrite(str(out.with_name(out.stem+f'_{k:03}.jpg')),canvas)
        if k%100==0:print('Render',out.stem,k,flush=True)
    cap.release();writer.release();import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(raw),'-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    cap=cv2.VideoCapture(str(out));fps=cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1000,1600,3);decoded+=1
    cap.release();assert decoded==n and fps==25;write(out.with_suffix('.verification.json'),{'decoded_frames':decoded,'fps':fps,'dimensions':[1600,1000]})
    print('Verified',out,decoded,'frames',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',default=str(OUT/'runs/v1_balanced_s29/filters/tracks.json'));p.add_argument('--output',default=str(OUT/'preliminary_v1.mp4'));args=p.parse_args()
    base=read(ROOT/'Dance_Tracking_Pilot/tracks.json')['roster'];after=read(args.candidate);render(base,after,ROOT/'Dance_Controlled_Test/source.mp4',args.output,read(ROOT/'Dance_Support_Training_Pilot/labels.json'))
