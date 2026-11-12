import json
import numpy as np
from ultralytics import YOLO
import infer
from infer import PILOT, read, image
from fresh_setup import OUT, CLIPS, crop_bounds
from compare import frame_predictions, make_track, time_series, smooth, match_boxes, pixel, error, measure

def main():
    infer.ROOT=OUT;infer.CLIPS=CLIPS
    infer.main()
    predictions=read(OUT/'predictions.json')
    model=YOLO(read(OUT/'training.json')['weights'])
    outputs={}
    for clip in CLIPS.values():
        for f in clip:
            boxes,_=frame_predictions(predictions[str(f)],.10)
            im=image(f); bounds=[crop_bounds(b) for b in boxes]
            points={}
            for start in range(0,len(boxes),12):
                group=bounds[start:start+12]
                results=model.predict([im[t:b,l:r] for l,t,r,b in group],imgsz=320,conf=.15,device=0,verbose=False)
                for j,(result,(l,t,r,b)) in enumerate(zip(results,group),start):
                    target=boxes[j,:4]-[l,t,l,t]
                    matches=match_boxes([target],result.boxes.xyxy.cpu().numpy(),.3)
                    if 0 not in matches:continue
                    p=result.keypoints.data.cpu().numpy()[matches[0],0]
                    if p[2]>=.5:points[j]=(p[:2]+[l,t]).tolist()
            outputs[str(f)]=points
            if f%25==0:print('Crop inference',f,flush=True)
    (OUT/'crop_predictions.json').write_text(json.dumps(outputs))
    from compare import stage, overlap
    all_series={};all_tracks={}
    for name,clip in CLIPS.items():
        tracks=make_track(clip,predictions); series=time_series(tracks)
        for tid,frames in tracks.items():
            raw={}
            for f,row in frames.items():
                boxes,_=frame_predictions(predictions[str(f)],.10)
                i=max(range(len(boxes)),key=lambda j:overlap(row['box_xyxy'],boxes[j]))
                p=outputs[str(f)].get(i)
                if p is not None:raw[f]=stage(p)
            series[tid]['crop_raw']=raw
            series[tid]['crop_smooth'],series[tid]['crop_estimated']=smooth(raw)
        all_series[name]=series;all_tracks[name]=tracks
    refs=read(OUT/'manual_references.json')
    modes=['box_smooth','guided_stable','crop_smooth']
    samples=[]
    for fs,people in refs.items():
        f=int(fs);name=next(n for n,c in CLIPS.items() if f in c)
        present=[(tid,rows[f]['box_xyxy']) for tid,rows in all_tracks[name].items() if f in rows]
        links=match_boxes([p['box'] for p in people],[p[1] for p in present],.35)
        for i,p in enumerate(people):
            sample={}
            if i in links:
                seq=all_series[name][present[links[i]][0]]
                for m in modes:
                    if f in seq[m]:sample[m]=error(pixel(seq[m][f]),p['point'])
            samples.append(sample)
    common=[s for s in samples if all(m in s for m in modes)]
    # Temporal roughness is a motion proxy, not ground-truth jitter in this moving scene.
    rough={m:[] for m in modes}
    for ss in all_series.values():
        for seq in ss.values():
            for f in seq['box_smooth']:
                if not all(all(k in seq[m] for k in (f-1,f,f+1)) for m in modes):continue
                for m in modes:rough[m].append(float(np.linalg.norm(seq[m][f+1]-2*seq[m][f]+seq[m][f-1])))
    report={'spot_check':{m:measure([s[m] for s in samples if m in s],len(samples)) for m in modes},
        'paired':{m:measure([s[m] for s in common],len(common)) for m in modes},
        'roughness':{m:{'samples':len(v),'median_second_difference':float(np.median(v)),
                       'p95_second_difference':float(np.percentile(v,95))} for m,v in rough.items()},
        'limitations':'Small visually placed contact sample, incomplete detector coverage, assisted training labels. Roughness includes genuine motion; no measured ID-switch rate or physical calibration.'}
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    # Preserve arrays as lists for rendering and the notebook.
    (OUT/'trajectories.json').write_text(json.dumps({'tracks':all_tracks,'series':all_series},
        default=lambda x:sorted(x) if isinstance(x,set) else x.tolist()))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
