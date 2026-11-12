import json
import subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
import nbformat
from nbclient import NotebookClient
from infer import image, read
from fresh_setup import OUT,CLIPS
from compare import SCALE
from replay import text,color

def render():
    data=read(OUT/'trajectories.json')
    panels=[('Smoothed box centers','box_smooth','box_estimated'),
            ('Frozen tracked hybrid','guided_stable','box_estimated'),
            ('New trained contact model','crop_smooth','crop_estimated')]
    temp=OUT/'preview_work.mp4'; target=OUT/'fresh_pilot.mp4'
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),25,(1600,900))
    for name,clip in CLIPS.items():
        tracks=data['tracks'][name]; series=data['series'][name]
        for f in clip:
            fs=str(f); im=image(f)
            for tid,seq in tracks.items():
                if fs not in seq:continue
                x1,y1,x2,y2=map(round,seq[fs]['box_xyxy'])
                cv2.rectangle(im,(x1,y1),(x2,y2),color(int(tid)),1)
                text(im,str(tid),x1,y1-3,.35,color(int(tid)))
            canvas=np.full((900,1600,3),(27,23,17),np.uint8)
            text(canvas,'DANCE / FRESH-SECTION PILOT',28,40,1.,width=2)
            text(canvas,f'{"FLOOR WORK & OVERLAP" if name=="overlap" else "CROWDED TRANSITION"} | {f/25:.2f}s | frame {f} | normal speed',28,70,.60)
            canvas[89:489,24:1576]=cv2.resize(im[280:610],(1552,400))
            for i,(title,mode,fill) in enumerate(panels):
                x=24+524*i;l=x+38;t=558;w=416;h=260
                text(canvas,title,x+12,530,.74,width=2)
                cv2.rectangle(canvas,(l,t),(l+w,t+h),(52,43,32),-1)
                for k in range(9):cv2.line(canvas,(l+round(w*k/8),t),(l+round(w*k/8),t+h),(88,73,56),1)
                for k in range(6):cv2.line(canvas,(l,t+round(h*k/5)),(l+w,t+round(h*k/5)),(88,73,56),1)
                count=0
                for tid,seq in series.items():
                    p=seq[mode].get(fs)
                    if p is None:continue
                    u,v=np.array(p)/SCALE
                    if not (0<=u<=1 and 0<=v<=1):continue
                    uncertain=f in seq[fill] or (mode=='guided_stable' and fs not in seq['ankles'])
                    cx,cy=l+round(w*u),t+round(h*v)
                    cv2.circle(canvas,(cx,cy),5,color(int(tid)),1 if uncertain else -1,cv2.LINE_AA)
                    count+=1
                text(canvas,f'{count} displayed dots',x+38,846,.48)
            text(canvas,'Same detector and IDs. Hollow: interpolated / box fallback. Floor contact and mat geometry remain estimates.',28,883,.53)
            writer.write(canvas)
            if f==4025:cv2.imwrite(str(OUT/'fresh_pilot.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
    writer.release()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(temp),
        '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    temp.unlink()

def notebook():
    m=read(OUT/'results.json')
    n=nbformat.v4.new_notebook()
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    n.cells=[
      md('# Dance positions: a fresh test\n\nCan a separate contact model improve the stage map when dancers overlap or move onto the floor? We compare three methods using the same pretrained detector and tracker.'),
      md('## What changed\n\nEarlier tests showed that raw ankle points helped with high kicks but were unstable during floor work. This experiment freezes the existing hybrid and adds a contact model trained on individual dancer crops. The comparison uses fresh footage at **2:37–2:42** and **3:28–3:32**.'),
      code('from pathlib import Path\nimport json\nfrom IPython.display import display, HTML, Video\nroot = Path.cwd()\nresults = json.loads((root / "results.json").read_text())\nprotocol = json.loads((root / "protocol.json").read_text())\nprint("Fresh test windows:", protocol["clips"])'),
      md('## Training\n\nWe reused the existing 48 training and 12 validation images. Each labeled dancer with a usable contact became a crop. The existing one-point YOLO model was adapted for 12 epochs at 320-pixel input, with the first 10 layers frozen and a learning rate of 0.0002. Validation selected the checkpoint. No fresh test frame entered training.\n\nThe labels are still assisted drafts. A crop may contain part of another dancer, so target identity can be ambiguous. High validation pose AP is not enough to show that a predicted point touches the floor.'),
      code('train_crops = len(list((root / "crops/images/train").glob("*.jpg")))\nval_crops = len(list((root / "crops/images/val").glob("*.jpg")))\nprint(f"Training crops: {train_crops}; validation crops: {val_crops}")'),
      md('## Fair comparison\n\nAll variants use pretrained YOLO boxes, ByteTrack IDs and the same draft mat transform. The box baseline smooths its bottom center. The frozen hybrid adds a persistent sideways ankle correction. The crop model predicts a contact within each detected person crop and then smooths those points.\n\nSmoothing uses neighboring frames, including a short look-ahead. This is an offline video experiment. A filled gap is an estimate, and an ankle prediction does not establish that a jumping dancer touches the floor.'),
      code('display(Video(str(root / "fresh_pilot.mp4"), embed=True, width=1100))'),
      md('## Accuracy spot-check\n\nWe placed 21 approximate reference contacts on four fresh frames before inspecting predictions. Only clear supporting feet or knee supports were selected. This is a small, selective check, not complete ground truth. The table counts missing estimates against coverage; the paired results use only contacts available to all three methods.'),
      code('names = {"box_smooth":"Smoothed boxes", "guided_stable":"Frozen hybrid", "crop_smooth":"Trained crop model"}\nrows = []\nfor key, label in names.items():\n    s = results["spot_check"][key]\n    p = results["paired"][key]\n    rows.append(f"<tr><td>{label}</td><td>{s[\'coverage\']}/{s[\'references\']}</td><td>{s[\'within_10px\']}</td><td>{p[\'median_px\']:.1f}</td><td>{p[\'median_map\']:.2f}%</td></tr>")\ndisplay(HTML("<table><tr><th>Method</th><th>Coverage</th><th>Within 10px</th><th>Paired median px</th><th>Paired map error</th></tr>" + "".join(rows) + "</table>"))'),
      md('## Smoothness and limitations\n\nThese dancers are moving, so frame-to-frame motion cannot be called jitter. We compare trajectory roughness using second differences on the same track samples. Lower roughness can also mean that genuine movement has been damped. The video is needed to judge that tradeoff.\n\nThe mat dimensions are unmeasured; map errors use percent of displayed mat width. Missing dancers remain the main limitation in crowded scenes. ID consistency has not been independently scored.'),
      code('for key, label in names.items():\n    s = results["roughness"][key]\n    print(f"{label}: p95 roughness {s[\'p95_second_difference\']:.3f}, {s[\'samples\']} shared samples")'),
      md('## Decision from this pilot\n\nThe frozen hybrid brought almost no benefit on these difficult sections. The crop model reduced median error on the 16 matched contacts from 15.0 to 11.7 pixels, but only 5 of 21 references were within 10 pixels, compared with 6 for the box baseline. Its motion was slightly rougher. This is not a clear replacement.\n\nAt source frame 4025, only five dancers have active tracks. The missing detections dominate the stage map. Keep smoothed box centers as the reference pipeline and improve detection coverage before committing to a separate contact network.\n\nFirst correct missed and duplicate dancer boxes in the existing dataset. Give each distinguishable dancer one box, even when their feet are hidden. Add a floor point only when its position can be defended, and record hidden or airborne cases separately. Prioritize overlap and low poses. Keep a new continuous window untouched for the next comparison. The teammate guide should wait until we agree on these results.'),
      md('## Reproduce this experiment\n\nThe scripts are in the neighboring `Dance_Floor_Contact` folder: `fresh_setup.py` prepares crops and trains; `fresh_run.py` runs inference and evaluation; `fresh_deliver.py` renders this replay and notebook. Saved JSON files preserve the split, references, model path and results.\n\nAI assistance was used for code, label proposals, visual reference placement and this report. Human review is still needed for the annotation ground truth.')]
    n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
    path=OUT/'Fresh_Dance_Pilot.ipynb'
    nbformat.write(n,path)
    NotebookClient(n,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
    nbformat.write(n,path)
    print(path)

if __name__=='__main__':render();notebook()
