"""Render an auditable controlled comparison; no manual corrections after frame zero."""
from collections import defaultdict
import json, subprocess, shutil, colorsys,sys
from pathlib import Path
import cv2
import numpy as np
from controlled import OUT,FIRST,COUNT
from compare import bottom,stage,smooth,pixel

INK=(237,239,245); MUTED=(168,181,197); BG=(23,20,17)
def color(i):
    if i>20:return (75,150,255)
    rgb=colorsys.hsv_to_rgb(((i-1)*.61803398875)%1,.65,1)
    return tuple(round(v*255) for v in rgb[::-1])
def text(im,s,x,y,size=.55,c=INK,t=1):
    cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,c,t,cv2.LINE_AA)
def series(rows,mask_anchor=False):
    raw=defaultdict(dict)
    for f,objects in enumerate(rows):
        for o in objects:raw[o['id']][f]=stage(o['mask_bottom'] if mask_anchor else bottom(o['box']))
    return {i:(seq,*smooth(seq)) for i,seq in raw.items()}
def grid(im):
    for u in np.linspace(0,100,9):
        a,b=pixel([u,0]),pixel([u,62.5]);cv2.line(im,tuple(a.astype(int)),tuple(b.astype(int)),(86,91,85),1,cv2.LINE_AA)
    for v in np.linspace(0,62.5,7):
        a,b=pixel([0,v]),pixel([100,v]);cv2.line(im,tuple(a.astype(int)),tuple(b.astype(int)),(86,91,85),1,cv2.LINE_AA)
def overlay(im,objects,mask=None):
    im=im.copy();grid(im)
    if mask is not None:
        for o in objects:
            sel=mask==o['id'];im[sel]=(im[sel]*.55+np.array(color(o['id']))*.45).astype(np.uint8)
    for o in objects:
        i=o['id'];x,y,x2,y2=map(int,o['box']);c=color(i)
        cv2.rectangle(im,(x,y),(x2,y2),c,1,cv2.LINE_AA)
        label=str(i) if i<=20 else 'N'+str(i-100)
        cv2.rectangle(im,(x,y-15),(x+max(19,len(label)*9),y),BG,-1)
        text(im,label,x+2,y-3,.39,c)
        cv2.circle(im,tuple(bottom(o['box']).astype(int)),3,c,-1)
    return im
def map_panel(canvas,x,y,seqs,f):
    w,h=650,406
    cv2.rectangle(canvas,(x,y),(x+w,y+h),(36,33,29),-1)
    for u in np.linspace(0,1,9):cv2.line(canvas,(x+round(u*w),y),(x+round(u*w),y+h),(69,66,60),1)
    for v in np.linspace(0,1,7):cv2.line(canvas,(x,y+round(v*h)),(x+w,y+round(v*h)),(69,66,60),1)
    text(canvas,'BACK',x+w//2-20,y-8,.4,MUTED);text(canvas,'FRONT',x+w//2-25,y+h+21,.4,MUTED)
    outside=0;visible=0
    def loc(p):return (round(x+p[0]/100*w),round(y+p[1]/62.5*h))
    def inside(p):return 0<=p[0]<=100 and 0<=p[1]<=62.5
    for i,(raw,sm,filled) in seqs.items():
        if f not in sm:continue
        p=sm[f]
        if not inside(p):outside+=1;continue
        visible+=1;c=color(i)
        for t in range(max(0,f-20),f):
            if t in sm and t+1 in sm and inside(sm[t]) and inside(sm[t+1]):
                cv2.line(canvas,loc(sm[t]),loc(sm[t+1]),tuple(int(v*.55) for v in c),1,cv2.LINE_AA)
        if f in raw and inside(raw[f]):cv2.circle(canvas,loc(raw[f]),2,(130,130,130),-1)
        pt=loc(p);cv2.circle(canvas,pt,6,c,2 if f in filled else -1,cv2.LINE_AA)
        text(canvas,str(i) if i<=20 else 'N'+str(i-100),pt[0]+8,pt[1]+4,.39,c)
    return visible,outside
def metrics(rows,seqs):
    counts=[len(r) for r in rows[1:]]
    seeded=[sum(o['id']<=20 for o in r) for r in rows[1:]]
    new=set(o['id'] for r in rows[1:] for o in r if o['id']>20)
    def motion(idx):
        steps=[]
        for raw,sm,filled in seqs.values():
            s=(raw,sm)[idx]
            for f in range(2,COUNT):
                if f in s and f-1 in s and f not in filled and f-1 not in filled:
                    steps.append(float(np.linalg.norm(s[f]-s[f-1])))
        return {'pairs':len(steps),'median':float(np.median(steps)),'p95':float(np.percentile(steps,95)),'over_3_percent_mat_width':sum(s>3 for s in steps)}
    return {'output_count_median':float(np.median(counts)),'output_count_min':min(counts),'output_count_max':max(counts),
      'initial_id_outputs_median':float(np.median(seeded)),'initial_id_outputs_final':seeded[-1],
      'new_ids_total':len(new),'raw_map_step':motion(0),'smoothed_map_step':motion(1),
      'note':'Output counts are not dancer recall; IDs are not checked identity correctness. Motion includes genuine dancing, so these are jump diagnostics, not ground-truth jitter scores.'}

def reference_check(allseqs):
    # Existing, approximate human contact marks, not a new held-out test set.
    path=OUT.parent/'Dance_CVAT_Pilot/full_clip_keyframes.txt'
    lines=[s.split() for s in path.read_text().splitlines() if s and not s.startswith('#')]
    frames=list(map(int,lines[0])); refs={}
    for row in lines[1:]:
        i=int(row[0][1:]);prev=None
        for f,value in zip(frames,row[1:]):
            if value=='=':value=prev
            prev=value
            if value[0] in 'hu':continue
            refs[i,f]=stage(list(map(float,value.split(','))))
    errors={}
    for n,seqs in allseqs.items():
        errors[n]={k:float(np.linalg.norm(seqs[k[0]][1][k[1]]-p)) for k,p in refs.items() if k[1]>0 and k[0] in seqs and k[1] in seqs[k[0]][1]}
    common=set.intersection(*(set(e) for e in errors.values()))
    result={'scope':'First 3 seconds only. Existing approximate visible-contact references, matched by initial ID. Initialization frame excluded. Not held-out; no manual identity validation of all predictions.', 'reference_count':sum(f>0 for i,f in refs),'common_predictions':len(common)}
    for n,e in errors.items():
        v=list(e.values());vcommon=[e[k] for k in common]
        result[n]={'predictions_at_reference':len(v),'median_error_percent_mat_width':float(np.median(v)),'p90_error_percent_mat_width':float(np.percentile(v,90)),'paired_median_error_percent_mat_width':float(np.median(vcommon))}
    return result

def main():
    alternative='--mask-anchor' in sys.argv
    suffix='_maskanchor' if alternative else ''
    rows={n:json.loads((OUT/f'{n}.json').read_text()) for n in ('yolo','sam')}
    assert all(len(v)==COUNT for v in rows.values())
    seqs={n:series(v,alternative and n=='sam') for n,v in rows.items()}
    stats={n:metrics(rows[n],seqs[n]) for n in rows}
    stats['approximate_contact_check']=reference_check(seqs)
    (OUT/f'metrics{suffix}.json').write_text(json.dumps(stats,indent=2))
    cap=cv2.VideoCapture(str(OUT/'source.mp4'))
    writer=cv2.VideoWriter(str(OUT/f'comparison{suffix}_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(1600,1000))
    audit=[]
    for f in range(COUNT):
        ok,im=cap.read();assert ok,f
        canvas=np.full((1000,1600,3),BG,np.uint8)
        text(canvas,'20 DANCERS / ONE STARTING FRAME / NO LATER CORRECTIONS',24,36,.72,INK,2)
        text(canvas,f'12-second controlled test   |   clip {f/25:05.2f}s   |   original {(FIRST+f)/25:.2f}s',24,63,.47,MUTED)
        for col,n in enumerate(('yolo','sam')):
            x=24+col*800;objects=rows[n][f]
            title='CROPPED YOLO + ByteTrack' if n=='yolo' else 'SAM 2.1 SMALL / mask propagation'
            text(canvas,title,x,100,.65,INK,2)
            retained=sum(o['id']<=20 for o in objects)
            text(canvas,f'{len(objects)} outputs  |  {retained}/20 initial IDs output',x,126,.48,MUTED)
            mask=cv2.imread(str(OUT/'masks'/f'{f:04}.png'),0) if n=='sam' else None
            annotated=overlay(im,objects,mask)
            canvas[145:407,x:x+752]=cv2.resize(annotated[285:635,140:1145],(752,262))
            map_title='STAGE MAP / mask bottom (99th percentile) + smoothing' if alternative and n=='sam' else 'STAGE MAP / box-bottom centre + smoothing'
            text(canvas,map_title,x,447,.46,INK)
            visible,outside=map_panel(canvas,x+51,481,seqs[n],f)
            text(canvas,f'{visible} map dots  |  {outside} outside mat',x+51,937,.46,MUTED)
        text(canvas,'Tiny grey dots: raw positions   |   Colour: smoothed   |   Hollow: interpolated gap (max 4 frames)   |   N: new YOLO ID',24,970,.43,MUTED)
        text(canvas,'Map dots are position proxies, not verified floor contacts. Draft mat calibration; dimensions are relative.',24,991,.42,MUTED)
        writer.write(canvas)
        if f in (0,75,150,225,299):
            cv2.imwrite(str(OUT/f'comparison{suffix}_{f:03}.jpg'),canvas)
            strips=[]
            for n in ('yolo','sam'):
                mask=cv2.imread(str(OUT/'masks'/f'{f:04}.png'),0) if n=='sam' else None
                strip=overlay(im,rows[n][f],mask)[300:600,170:1160].copy()
                text(strip,f'{n.upper()} +{f/25:.2f}s',8,22,.6,INK,2);strips.append(strip)
            audit.append(np.vstack(strips))
    cap.release();writer.release()
    for i,a in enumerate(audit):cv2.imwrite(str(OUT/f'audit{suffix}_{i}.jpg'),a)
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-y','-i',str(OUT/f'comparison{suffix}_raw.mp4'),'-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'comparison{suffix}.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
