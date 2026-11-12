"""Hold the detector fixed, compare floor estimators, and measure temporal smoothing."""

from collections import defaultdict
import json
import math
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from infer import CLIPS, PILOT, REVIEW, ROOT, read

H = np.asarray(read(REVIEW / 'calibration_draft.json')['pixel_to_relative_mat_homography'])
H_INV = np.linalg.inv(H)
SCALE = np.array([100., 100./1.6])  # Mat display aspect only, no physical dimensions.


def transform(point, matrix):
    vec = matrix @ [*point, 1.]
    return vec[:2] / vec[2]


def stage(point):
    return transform(point,H)*SCALE


def pixel(point):
    return transform(np.asarray(point)/SCALE,H_INV)


def bottom(box):
    return np.array([(box[0]+box[2])/2, box[3]])


def overlap(a,b):
    wh = np.maximum(0, np.minimum(a[2:4], b[2:4]) - np.maximum(a[:2], b[:2]))
    cross = np.prod(wh)
    return float(cross / max(np.prod(a[2:4]-a[:2])+np.prod(b[2:4]-b[:2])-cross,1e-6))


def assign(quality, minimum):
    if not quality.size:
        return {}
    a,b = linear_sum_assignment(-np.where(quality>=minimum, 10+quality, 0))
    return {int(i):int(j) for i,j in zip(a,b) if quality[i,j]>=minimum}


def match_boxes(a,b,minimum=.50):
    return assign(np.array([[overlap(np.asarray(x),np.asarray(y)) for y in b] for x in a]),minimum)


def ankles(kp, box):
    x1,y1,x2,y2 = box[:4]
    h,w = y2-y1,x2-x1
    pts = []
    for knee, ankle in ((13,15),(14,16)):
        x,y,c = kp[ankle]
        if c < .44 or not (x1-.06*w<=x<=x2+.06*w and y1+.48*h<=y<=y2+.12*h):
            continue
        # An ankle is above the sole. Continue down slightly, with a tightly
        # limited horizontal correction from the lower leg direction.
        kx,ky,kc = kp[knee]
        dx = np.clip((x-kx)*.12,-.035*h,.035*h) if kc >= .35 and y-ky > .10*h else 0.
        pts.append((x+dx,y+.035*h,c))
    if not pts:
        return None
    pts.sort(key=lambda p:p[1],reverse=True)
    if len(pts)==2 and pts[0][1]-pts[1][1] < max(5.,.07*h):
        return np.average([p[:2] for p in pts],axis=0,weights=[p[2] for p in pts])
    return np.asarray(pts[0][:2])


def associate_pose(boxes, poses, mode):
    # Tiles sometimes see the same person. Dedupe high-overlap proposals before
    # one-to-one matching, so two detector boxes cannot borrow one skeleton.
    candidates = []
    for pose in sorted(poses,key=lambda p:p['score'],reverse=True):
        box = np.asarray(pose['box'])
        kp = np.asarray(pose['kp'])
        point = ankles(kp,box) if mode=='ankles' else kp[0,:2] if kp[0,2]>=.5 else None
        if point is None or any(overlap(box,previous[0])>=.78 for previous in candidates):
            continue
        candidates.append((box,point))
    links = match_boxes(boxes,[p[0] for p in candidates],minimum=.36)
    result = {}
    for i,j in links.items():
        point = candidates[j][1]
        uv = stage(point)/SCALE
        if np.all(uv>=-.035) and np.all(uv<=1.035):
            result[i] = point
    return result


def measure(errors, total):
    a = np.asarray(errors)
    return {'coverage':len(a),'references':total,
            'median_px':float(np.median(a[:,0])) if len(a) else None,
            'p90_px':float(np.percentile(a[:,0],90)) if len(a) else None,
            'median_map':float(np.median(a[:,1])) if len(a) else None,
            'p90_map':float(np.percentile(a[:,1],90)) if len(a) else None,
            'within_10px':int(np.sum(a[:,0]<=10.)) if len(a) else 0}


def error(estimate, ref):
    return [float(np.linalg.norm(estimate-ref)), float(np.linalg.norm(stage(estimate)-stage(ref)))]


RAW = ['box','ankles','learned','guided']


def guided(box, ankle):
    """Exploratory variant: correct broad poses laterally, retain box depth."""
    estimate = bottom(box)
    if ankle is None:
        return estimate
    w,h = box[2]-box[0], max(box[3]-box[1],1.)
    width_weight = np.clip((w/h-.48)/.32,0.,1.)
    depth_agreement = np.clip(1.-max(abs(ankle[1]-estimate[1])-.035*h,0.)/(.10*h),0.,1.)
    estimate[0] += width_weight*depth_agreement*(ankle[0]-estimate[0])
    return estimate


def frame_predictions(d, confidence):
    boxes = np.asarray(d['boxes'],dtype=np.float32).reshape(-1,6)
    boxes = boxes[boxes[:,4]>=confidence]
    points = {'box':{i:bottom(b) for i,b in enumerate(boxes)},
              'ankles':associate_pose(boxes,d['pose'],'ankles'),
              'learned':associate_pose(boxes,d['learned'],'learned')}
    points['guided'] = {i:guided(box,points['ankles'].get(i)) for i,box in enumerate(boxes)}
    return boxes, points


def stills(predictions):
    labels = read(PILOT / 'pilot_labels.json')
    refs = read(PILOT / 'manual_test_contacts.json')
    data = defaultdict(dict)
    for row in read(PILOT / 'split_manifest.json'):
        f, name, split = row['source_frame'],row['filename'],row['split']
        if split not in ('val','test'):
            continue
        boxes, points = frame_predictions(predictions[str(f)],.25)
        matches = match_boxes([r['box'] for r in labels[name]],boxes)
        samples = refs.get(name,{}) if split=='test' else {
            str(i+1):o['floor'] for i,o in enumerate(labels[name]) if o.get('anchor_source')=='visual_pick' and o['floor']}
        for i, ref in samples.items():
            j = matches.get(int(i)-1)
            data[split][(f,i)] = {mode:error(points[mode][j],ref) for mode in RAW if j in points[mode]}
    result = {}
    for split, samples in data.items():
        result[split] = {mode:measure([r[mode] for r in samples.values() if mode in r],len(samples)) for mode in RAW}
        common = [r for r in samples.values() if all(k in r for k in ('box','ankles'))]
        result[split]['paired_box_ankles'] = {mode:measure([r[mode] for r in common],len(common)) for mode in ('box','ankles')}
    return result


def make_track(clip, predictions, stride=1):
    t = BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.10,
                                    new_track_thresh=.25,track_buffer=25,match_thresh=.8,fuse_score=True))
    tracks = defaultdict(dict)
    for f in clip:
        if (f-clip.start)%stride:
            t.update(Boxes(np.empty((0,6),np.float32),(720,1280)))
            continue
        boxes, points = frame_predictions(predictions[str(f)],.10)
        matched = t.update(Boxes(boxes,(720,1280)))
        for row in matched:
            tid,i = int(row[4]),int(row[-1])
            tracks[tid][f] = {'box_xyxy':boxes[i,:4], **{m:stage(p[i]) for m,p in points.items() if i in p}}
    return tracks


def smooth(series, radius=3, max_missing=4):
    # Bounded look-ahead for recorded footage. Never extrapolate past a track.
    filled = dict(series)
    observed = sorted(series)
    for a,b in zip(observed,observed[1:]):
        gap = b-a-1
        if 0<gap<=max_missing and np.linalg.norm(series[b]-series[a]) <= 1.+.65*gap:
            for t in range(a+1,b):
                filled[t] = series[a]+(series[b]-series[a])*(t-a)/(b-a)
    out = {}
    for f,point in filled.items():
        neighbors = [k for k in range(f-radius,f+radius+1) if k in filled]
        # No averaging across an unfilled gap.
        a=b=f
        while a-1 in filled and f-a<radius: a-=1
        while b+1 in filled and b-f<radius: b+=1
        neighbors = [k for k in neighbors if a<=k<=b]
        if len(neighbors)<4 or f not in range(a+1,b):
            out[f]=point
            continue
        tau = np.asarray(neighbors,dtype=float)-f
        y = np.stack([filled[t] for t in neighbors])
        X = np.column_stack((np.ones(len(tau)),tau))
        w = np.exp(-.5*(tau/1.7)**2)*np.array([1. if t in series else .40 for t in neighbors])
        coefs = np.linalg.lstsq(X*w[:,None],y*w[:,None],rcond=None)[0]
        residual = np.linalg.norm(y-X@coefs,axis=1)
        w *= np.minimum(1.,.65/np.maximum(residual,.001))
        out[f] = np.linalg.lstsq(X*w[:,None],y*w[:,None],rcond=None)[0][0]
    return out, set(filled)-set(observed)


def time_series(tracks):
    result = {}
    for tid,frames in tracks.items():
        series = {mode:{f:p[mode] for f,p in frames.items() if mode in p} for mode in RAW}
        series['ankles_smooth'],series['ankles_estimated'] = smooth(series['ankles'])
        series['box_smooth'],series['box_estimated'] = smooth(series['box'])
        series['guided_smooth'],series['guided_estimated'] = smooth(series['guided'])
        # Require the ankle correction to persist across several frames. Smooth
        # the correction independently so fast common motion is preserved.
        offset = {f:p[0]-series['box'][f][0] for f,p in series['guided'].items()}
        median = {f:float(np.median([offset[t] for t in range(f-4,f+5) if t in offset]))
                  for f in offset}
        series['guided_stable'] = {}
        for f,p in series['box_smooth'].items():
            votes = [median[t] for t in range(f-2,f+3) if t in median]
            correction = float(np.mean(votes)) if len(votes)>=3 else 0.
            series['guided_stable'][f] = p + [correction,0.]
        result[tid] = series
    return result


def review_references():
    lines = [l.split() for l in (REVIEW/'full_clip_keyframes.txt').read_text().splitlines() if l and not l.startswith('#')]
    keys = [4420+int(x) for x in lines[0]]
    refs = {}
    stationary = {}
    for name,*vals in lines[1:]:
        points, r = {}, None
        first = None
        for f,v in zip(keys,vals):
            if v != '=':
                r = np.array(list(map(float,v.split(',')))) if v[0].isdigit() else None
            if first is None and v != '=':
                first = f
            elif name not in stationary and v!='=':
                if f-first>=30 and points.get(first) is not None:
                    stationary[name] = (first,f-5,points[first])
                first = f
            if r is not None:
                points[f]=r
        refs[name]=points
    return refs,stationary


def attach_references(tracks, refs):
    # Assignment uses the tracked rectangle, identically for all anchor methods.
    by_frame = defaultdict(dict)
    for name,points in refs.items():
        for f,p in points.items(): by_frame[f][name]=p
    out = {}
    for f,people in by_frame.items():
        observed = [(tid,seq[f]['box_xyxy']) for tid,seq in tracks.items() if f in seq]
        quality=[]
        for p in people.values():
            scores=[]
            for _,b in observed:
                delta = np.abs(p-bottom(b))
                dx=max(b[0]-p[0],0,p[0]-b[2])
                cost=math.hypot(delta[0]*.48+dx,delta[1]*1.25)
                scores.append(1-cost/35)
            quality.append(scores)
        for i,j in assign(np.asarray(quality),.05).items():
            name=list(people)[i]
            out[name,f] = observed[j][0]
    return out


TEMP = ['box','ankles','box_smooth','ankles_smooth','guided','guided_smooth','guided_stable']


def clip_scores(series, reference_tracks, refs):
    data={}
    for name,points in refs.items():
        for f,p in points.items():
            tid=reference_tracks.get((name,f))
            data[name,f]={}
            if tid not in series:
                continue
            for mode in TEMP:
                estimate=series[tid][mode].get(f)
                if estimate is not None:
                    data[name,f][mode]=error(pixel(estimate),p)
    common=[row for row in data.values() if all(m in row for m in TEMP)]
    return {'all':{m:measure([x[m] for x in data.values() if m in x],len(data)) for m in TEMP},
            'paired':{m:measure([x[m] for x in common],len(common)) for m in TEMP}}


def jitter(series, reference_tracks, stationary):
    pairs={m:[] for m in TEMP}
    tracks_used={}
    for name,(start,stop,ref) in stationary.items():
        assignments=[reference_tracks.get((name,f)) for f in range(start,stop+1,5)]
        assigned=[t for t in assignments if t is not None]
        if not assigned:
            continue
        tid=max(set(assigned),key=assigned.count)
        if assigned.count(tid)<max(3,len(assignments)*.55):
            continue
        this=series[tid]
        count=0
        for f in range(start+1,stop+1):
            if any(f not in this[m] or f-1 not in this[m] for m in TEMP):
                continue
            count+=1
            for mode in TEMP:
                pairs[mode].append(float(np.linalg.norm(this[mode][f]-this[mode][f-1])))
        if count: tracks_used[name]={'track_id':tid,'pairs':count,'start':start,'stop':stop}
    return {'tracks':tracks_used,'samples':len(pairs['box']),
            'per_frame_map_step':{m:{'median':float(np.median(v)),'p95':float(np.percentile(v,95))}
                                  for m,v in pairs.items() if v}}


def sparse_score(dense, sparse, reference_tracks, refs):
    # Map sampled tracker IDs to the dense reference track via their shared boxes,
    # then score at the same visual keyframes including frames the detector skipped.
    votes=defaultdict(lambda:defaultdict(int))
    for sid,s in sparse.items():
        for f,row in s.items():
            for tid,d in dense.items():
                if f in d and overlap(row['box_xyxy'],d[f]['box_xyxy'])>.95:
                    votes[tid][sid]+=1
    translated={k:max(v,key=v.get) for k,v in votes.items()}
    return clip_scores(time_series(sparse),{k:translated.get(v) for k,v in reference_tracks.items()},refs)


def main():
    predictions=read(ROOT/'predictions.json')
    refs,stationary=review_references()
    dense=make_track(CLIPS['reviewed'],predictions)
    tracks=time_series(dense)
    links=attach_references(dense,refs)
    sparse=make_track(CLIPS['reviewed'],predictions,stride=3)
    report={'stills':stills(predictions),'reviewed_clip':clip_scores(tracks,links,refs),
            'stationary_jitter':jitter(tracks,links,stationary),
            'detect_every_3rd':sparse_score(dense,sparse,links,refs),
            'map_units':'percent of the displayed mat width; assumed 1.6 width-to-depth display ratio, no meters',
            'note':'Reference positions are visual annotations. Jitter uses identical pairs of consecutive observed contacts on approximately planted dancers. The guided blend was designed after seeing the initial results and is exploratory.'}
    (ROOT/'floor_contact_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
