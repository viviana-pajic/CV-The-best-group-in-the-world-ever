"""Produce a synchronized, slowed-down replay of the stage map variants."""

import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

from compare import make_track, time_series, SCALE
from infer import CLIPS, ROOT, image, read

INK=(229,239,244)
MUTED=(161,186,199)
BG=(27,23,17)
PALETTE=[(255,200,82),(181,229,83),(98,210,255),(243,129,173),(160,199,255),
         (192,137,255),(245,235,132),(117,234,176),(255,175,178),(198,228,246)]
PANELS=[('Box bottom','Raw baseline','box',None),
        ('Ankle contact','Raw pose estimate','ankles',None),
        ('Tracked box','Smoothed box bottom','box_smooth','box_estimated'),
        ('Tracked hybrid','Persistent ankle correction','guided_stable','box_estimated')]


def text(im,s,x,y,size=.56,color=INK,width=1):
    cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,width,cv2.LINE_AA)


def color(ident):
    return PALETTE[(ident*7+ident//10)%len(PALETTE)]


def panel(canvas,tracks,f,i,title,sub,key,imputed):
    x=18+i*396
    cv2.rectangle(canvas,(x,365),(x+376,695),(46,36,26),-1)
    text(canvas,title,x+16,398,.76,INK,2)
    text(canvas,sub,x+16,422,.49,MUTED)
    l,t,w,h=x+22,447,332,207
    cv2.rectangle(canvas,(l,t),(l+w,t+h),(57,51,40),-1)
    for k in range(9):
        u=l+round(w*k/8)
        cv2.line(canvas,(u,t),(u,t+h),(92,81,64),1,cv2.LINE_AA)
    for k in range(6):
        v=t+round(h*k/5)
        cv2.line(canvas,(l,v),(l+w,v),(92,81,64),1,cv2.LINE_AA)
    count=0
    for tid,series in tracks.items():
        if f not in series[key]:
            continue
        u,v=series[key][f]/SCALE
        if not (0<=u<=1 and 0<=v<=1):
            continue
        px,py=l+round(u*w),t+round(v*h)
        inferred=bool((imputed and f in series[imputed]) or (key=='guided_stable' and f not in series['ankles']))
        cv2.circle(canvas,(px,py),5,color(tid),1 if inferred else -1,cv2.LINE_AA)
        count+=1
    text(canvas,f'{count} dots',x+16,679,.46,MUTED)
    text(canvas,'front edge',x+265,679,.43,MUTED)


def render():
    cache=read(ROOT/'predictions.json')
    temp=ROOT/'floor_contact_replay_work.mp4'
    target=ROOT/'floor_contact_replay.mp4'
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),15,(1600,746))
    if not writer.isOpened():
        raise RuntimeError('Video writer unavailable')
    for name,clip in CLIPS.items():
        dense=make_track(clip,cache)
        tracks=time_series(dense)
        for f in clip:
            canvas=np.full((746,1600,3),BG,dtype=np.uint8)
            text(canvas,'DANCE  /  FLOOR CONTACT TEST',24,42,1.08,INK,2)
            text(canvas,'Same pretrained YOLO detections. Same track IDs and mat projection in all panels.',24,69,.58,MUTED)
            show=image(f)
            for tid,seq in dense.items():
                if f not in seq:
                    continue
                x1,y1,x2,y2=map(round,seq[f]['box_xyxy'])
                cv2.rectangle(show,(x1,y1),(x2,y2),color(tid),1,cv2.LINE_AA)
            show=cv2.resize(show[255:625],(910,263),interpolation=cv2.INTER_AREA)
            canvas[84:347,24:934]=show
            text(canvas,'01 / PLANTED & OCCLUDED' if name=='reviewed' else '02 / TURNS & HIGH KICKS',960,126,.77,INK,2)
            text(canvas,f'Source frame {f}    {f/25:.2f}s',960,157,.60,MUTED)
            text(canvas,'Playback at 0.6x speed',960,188,.57,MUTED)
            text(canvas,'Same color follows the same track.',960,246,.57,MUTED)
            text(canvas,'Hollow dot: gap fill or no reliable ankle cue.',960,273,.54,MUTED)
            text(canvas,'Ankles panel leaves uncertain contacts out.',960,300,.54,MUTED)
            for i,(title,sub,key,imputed) in enumerate(PANELS):
                panel(canvas,tracks,f,i,title,sub,key,imputed)
            text(canvas,'Draft mat geometry; map units are relative. A dot is an estimate, not proof of floor contact.',24,726,.53,MUTED)
            writer.write(canvas)
            if f==4568:
                cv2.imwrite(str(ROOT/'floor_contact_replay.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
    writer.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(temp),
                    '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    temp.unlink()
    print(target)


if __name__=='__main__':
    render()
