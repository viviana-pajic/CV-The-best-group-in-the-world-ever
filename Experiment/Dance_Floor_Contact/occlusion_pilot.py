"""Fixed-detector tracking experiment. Later annotations are used only by evaluate()."""
import json, time, hashlib, argparse
from pathlib import Path
from collections import deque, Counter
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from compare import stage, bottom, overlap, match_boxes, smooth
from corrected_setup import OUT as PREV, SOURCE, VAL, TEST, read

OUT=PREV.parent/'Dance_Tracking_Pilot'
OUT.mkdir(exist_ok=True)
CFG=dict(high_conf=.25,low_conf=.10,ground_weight=.55,ground_gate=8.,
         max_prediction_frames=25,identity_memory_frames=75,map_alpha=.35,
         map_jump=2.,map_persistence=3,roster=20,fps=25,body_rescue=False)

def save(name,data):
    def enc(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    (OUT/name).write_text(json.dumps(data,default=enc,indent=2),encoding='utf-8')

def inside(p):return p is not None and np.isfinite(p).all() and 0<=p[0]<=100 and 0<=p[1]<=62.5
def center(b):return (b[:2]+b[2:])/2
def norm(v,limit):
    n=np.linalg.norm(v)
    return v*min(1,limit/max(n,1e-9))

class Dancer:
    def __init__(self,o):
        self.id=o['id'];self.box=np.array(o['box'],float);self.last=0
        self.ground=stage(bottom(self.box));self.v=np.zeros(2);self.iv=np.zeros(2)
        self.history=deque([(0,self.ground.copy(),center(self.box))],maxlen=8)
        self.map=self.ground.copy() if inside(self.ground) else None
        self.map_last=0 if self.map is not None else -1000
        self.pending=[];self.point_history=deque(maxlen=3)

    def predicted(self,k):
        # Damped, bounded motion: no unbounded Kalman extrapolation through occlusion.
        age=k-self.last;dt=sum(.92**j for j in range(min(age,25)))
        return self.box+np.tile(self.iv*dt,2),self.ground+self.v*dt

    def update(self,b,k):
        p=stage(bottom(b));dt=k-self.last
        if dt<=5:
            self.history.append((k,p.copy(),center(b)))
            if len(self.history)>2:
                a=self.history[0];d=max(k-a[0],1)
                self.v=.75*self.v+.25*norm((p-a[1])/d,.7)
                self.iv=.75*self.iv+.25*norm((center(b)-a[2])/d,6.)
        else:
            self.v*=0;self.iv*=0;self.history.clear();self.history.append((k,p.copy(),center(b)))
        self.box=b.copy();self.ground=p;self.last=k

def associate(tracks,det,k,config):
    matches={};unused=set(range(len(det)))
    for threshold in [config['high_conf'],config['low_conf']]:
        ids=[i for i,t in enumerate(tracks) if i not in matches and k-t.last<=config['identity_memory_frames']]
        candidates=[j for j in sorted(unused) if det[j]['score']>=threshold]
        if not ids or not candidates:continue
        # Private dummy columns allow every identity to remain unmatched.
        cost=np.full((len(ids),len(candidates)+len(ids)),.96)
        for ii,i in enumerate(ids):
            t=tracks[i];pb,pg=t.predicted(k);age=k-t.last
            height=max(45,pb[3]-pb[1]);width=max(30,pb[2]-pb[0])
            gate=config['ground_gate']+min(age,25)*.10
            for jj,j in enumerate(candidates):
                b=np.array(det[j]['box']);g=stage(bottom(b));gd=np.linalg.norm(g-pg)
                dc=center(b)-center(pb)
                cd=np.linalg.norm(dc/np.array([max(width,60),max(height,90)]))
                iou=overlap(pb,b)
                rescue=config['body_rescue'] and age<=10 and iou>.30 and cd<.65
                if (gd>gate and not rescue) or cd>1.3 or abs(dc[0])>110:cost[ii,jj]=99;continue
                ground_cost=min(gd/5.,2.)
                c=config['ground_weight']*ground_cost+.25*(1-iou)+(1-config['ground_weight']-.25)*min(cd,2)
                if rescue:c=min(c,.55*(1-iou)+.45*cd)
                c+=.06*(1-det[j]['score'])
                if age>75 and (det[j]['score']<.35 or gd>6 or c>.60):c=99
                if threshold==config['low_conf'] and (c>.70 or age>10):c=99
                cost[ii,jj]=c
        rr,cc=linear_sum_assignment(cost)
        for ii,jj in zip(rr,cc):
            if jj<len(candidates) and cost[ii,jj]<.96:
                matches[ids[ii]]=candidates[jj];unused.remove(candidates[jj])
    return matches,unused

def run_roster(preds,seeds,config):
    tracks=[Dancer(o) for o in seeds];rows=[];diagnostics=[]
    for k in range(300):
        det=[o for o in preds[str(k)] if o['score']>=config['low_conf']]
        links,unused=associate(tracks,det,k,config) if k else ({},set())
        active=[]
        for i,t in enumerate(tracks):
            observed=k==0 or i in links
            b=t.box.copy() if k==0 else np.array(det[links[i]]['box']) if observed else t.predicted(k)[0]
            if observed and k:t.update(b,k)
            age=k-t.last
            state='observed' if observed else 'predicted' if age<=config['max_prediction_frames'] else 'unknown'
            r=dict(id=t.id,box=b.tolist(),status=state,age_frames=age,score=det[links[i]]['score'] if k and observed else None)
            # Detect a plausible occluder at the bottom-center proxy; this is a heuristic, not a visibility classifier.
            foot=bottom(b);occluders=[]
            if observed:
                for oi,oj in links.items():
                    if oi==i:continue
                    other=np.array(det[oj]['box'])
                    if other[0]+2<foot[0]<other[2]-2 and other[1]<foot[1]<other[3]-8:
                        occluders.append(tracks[oi].id)
            p=stage(foot);reason='box_floor_proxy'
            valid=observed and inside(p) and not occluders
            if valid:
                # A one-frame partial box must not pull a dot to another row of the mat.
                if t.map is not None and k-t.map_last<25 and np.linalg.norm(p-t.map)>config['map_jump']:
                    if t.pending and np.linalg.norm(p-t.pending[-1])>1.5:t.pending=[]
                    t.pending.append(p.copy())
                    valid=len(t.pending)>=config['map_persistence'];reason='unstable_bottom' if not valid else 'box_floor_proxy'
                else:t.pending=[]
            elif observed:reason='possible_occlusion' if occluders else 'off_mat_proxy'
            else:reason='person_not_detected'
            if valid:
                t.point_history.append(p.copy());measurement=np.median(t.point_history,axis=0)
                if t.map is None or k-t.map_last>25:t.map=measurement
                else:t.map=t.map+config['map_alpha']*norm(measurement-t.map,2.5)
                t.map_last=k
                r.update(map=t.map.copy(),map_status='proxy',raw_map=p)
            elif t.map is not None and k-t.map_last<=config['max_prediction_frames']:
                # Conservative zero-velocity floor forecast. A hidden body does not imply feet are visible.
                r.update(map=t.map.copy(),map_status='estimated',uncertainty=.5+.10*(k-t.map_last))
            else:r.update(map=None,map_status='unknown')
            r.update(map_age_frames=k-t.map_last,reason=reason,possible_occluders=occluders)
            active.append(r)
        rows.append(active)
        diagnostics.append({'frame':k,'unassigned_candidates':len(unused),'unassigned_high_confidence_boxes':[det[j]['box'] for j in sorted(unused) if det[j]['score']>=config['high_conf']], 'observed':sum(r['status']=='observed' for r in active),'predicted':sum(r['status']=='predicted' for r in active),'unknown':sum(r['status']=='unknown' for r in active)})
    return {'frames':rows,'diagnostics':diagnostics,'config':config}

def run_ocsort(preds,seeds):
    from ultralytics.trackers.oc_sort import OCSORT
    from ultralytics.engine.results import Boxes
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.10,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True,delta_t=3,inertia=.2,use_byte=True)
    tracker=OCSORT(args);tracker.update(Boxes(np.array([o['box']+[.99,0] for o in seeds],np.float32),(720,1280)))
    mapping={t.track_id:o['id'] for t,o in zip(tracker.tracked_stracks,seeds)};frames=[]
    for k in range(300):
        if k==0:frames.append([{'id':o['id'],'box':o['box']} for o in seeds]);continue
        arr=np.array([o['box']+[o['score'],0] for o in preds[str(k)] if o['score']>=.1],np.float32).reshape(-1,6)
        rr=tracker.update(Boxes(arr,(720,1280)));row=[]
        for a in rr:
            tid=int(a[4])
            if tid not in mapping:mapping[tid]=101+len(mapping)-20
            row.append({'id':mapping[tid],'box':a[:4].tolist()})
        frames.append(row)
    return {'frames':frames}

def evaluate(data,labels,frames):
    total=matched=correct=wrong=predictions=refpoints=position_good=identity_position_good=point_matched=point_emitted=0
    predicted_matches=predicted_correct=hidden_total=hidden_as_observed=0
    errors=[];observations={};hidden_support=flagged_hidden=visible_support=flagged_visible=0
    for k in frames:
        gt=[o for o in labels[str(k)] if not o.get('outside')]
        active=[o for o in data['frames'][k] if o.get('status','observed')=='observed']
        links=match_boxes([o['box'] for o in gt],[o['box'] for o in active],.5)
        total+=len(gt);predictions+=len(active);matched+=len(links)
        refpoints+=sum(o['floor'] is not None for o in gt)
        point_emitted+=sum(o.get('map') is not None for o in active)
        for i,j in links.items():
            a,b=gt[i],active[j];same=a['id']==b['id'];correct+=same;wrong+=not same
            observations.setdefault(a['id'],[]).append((k,b['id']))
            if a['floor'] is None:
                hidden_support+=1;flagged_hidden+=b.get('map_status')!='proxy'
            else:
                visible_support+=1;flagged_visible+=b.get('map_status')!='proxy'
            if a['floor'] is not None and b.get('map') is not None:
                e=np.linalg.norm(np.array(b['map'])-stage(a['floor']));errors.append(e);point_matched+=1
                position_good+=e<=3;identity_position_good+=e<=3 and same
        hidden=[o for o in labels[str(k)] if o.get('outside')]
        hidden_total+=len(hidden)
        hidden_as_observed+=sum(any(r['id']==o['id'] for r in active) for o in hidden)
        predicted=[o for o in data['frames'][k] if o.get('status')=='predicted']
        plinks=match_boxes([o['box'] for o in gt],[o['box'] for o in predicted],.5)
        predicted_matches+=len(plinks);predicted_correct+=sum(gt[i]['id']==predicted[j]['id'] for i,j in plinks.items())
    pairs=[(a,b) for seq in observations.values() for a,b in zip(seq,seq[1:]) if b[0]-a[0]<=20]
    return dict(reference_boxes=total,observed_boxes=predictions,matched_boxes=matched,precision=matched/max(predictions,1),recall=matched/max(total,1),correct_initial_ids=correct,wrong_initial_ids=wrong,correct_id_fraction_of_matches=correct/max(matched,1),correct_id_recall=correct/max(total,1),identity_pairs=len(pairs),id_changes=sum(a[1]!=b[1] for a,b in pairs),visible_floor_references=refpoints,matched_output_floor_points=point_matched,floor_points_emitted_observed_tracks=point_emitted,floor_correct_within_3pct=position_good,floor_and_initial_id_correct=identity_position_good,median_floor_error_pct=float(np.median(errors)) if errors else None,p90_floor_error_pct=float(np.percentile(errors,90)) if errors else None,predicted_boxes_matching_visible_person=predicted_matches,predicted_boxes_with_correct_initial_id=predicted_correct,fully_hidden_id_records=hidden_total,fully_hidden_ids_claimed_observed=hidden_as_observed,matched_hidden_supports=hidden_support,hidden_supports_flagged_uncertain=flagged_hidden,matched_visible_supports=visible_support,visible_supports_flagged_uncertain=flagged_visible)

def attach_map(data):
    raw={}
    for k,row in enumerate(data['frames']):
        for o in row:
            p=stage(bottom(np.array(o['box'])))
            if inside(p):raw.setdefault(o['id'],{})[k]=p
    for ident,seq in raw.items():
        sm,filled=smooth(seq)
        for k,p in sm.items():
            for o in data['frames'][k]:
                if o['id']==ident:o.update(map=p,map_status='estimated' if k in filled else 'proxy',status='observed')
    return data

def main():
    labels=read(PREV/'labels.json');preds=read(PREV/'predictions_detect.json');seeds=labels['0']
    start=time.perf_counter()
    baseline=attach_map(read(PREV/'tracks.json')['tuned_box'])
    oc=attach_map(run_ocsort(preds,seeds))
    # Validation-only comparison: three weights, with/without body-overlap rescue and longer identity memory.
    variants=[];runs=[]
    for rescue in [False,True]:
        for weight in [.35,.55,.70]:
            cfg={**CFG,'ground_weight':weight,'body_rescue':rescue,'identity_memory_frames':300 if rescue else 75}
            data=run_roster(preds,seeds,cfg);score=evaluate(data,labels,VAL)
            variants.append({'config':cfg,'validation':score});runs.append(data)
    index=max(range(len(runs)),key=lambda i:(variants[i]['validation']['correct_initial_ids'],variants[i]['validation']['matched_boxes']))
    selected=runs[index]
    save('selection.json',{'rule':'Maximize correct original-ID matches at IoU .5 on validation frames only; tie-break matched boxes. All six configurations retained. This is iterative development on a previously reviewed clip, not a fresh independent test.','variants':variants,'selected_index':index})
    save('settings.json',selected['config'])
    save('tracks.json',{'bytetrack':baseline,'ocsort':oc,'roster':selected})
    metrics={split:{name:evaluate(data,labels,frames) for name,data in [('bytetrack',baseline),('ocsort',oc),('roster',selected)]} for split,frames in [('validation',VAL),('development_test',TEST)]}
    save('metrics.json',metrics)
    save('provenance.json',{'source':str(SOURCE/'source.mp4'),'detector_cache':str(PREV/'predictions_detect.json'),'detector_sha256':hashlib.sha256((PREV/'predictions_detect.json').read_bytes()).hexdigest(),'labels_sha256':hashlib.sha256((PREV/'labels.json').read_bytes()).hexdigest(),'frames':300,'fps':25,'initialization':'Same twenty reviewed first-frame boxes for every tracker; no manual floor points or later labels used by tracking.','scope':'Same development clip previously used in detector training and review. Final block was excluded from detector training but is not an untouched benchmark.','seconds_tracking_and_evaluation':time.perf_counter()-start,'known_limitations':['Fixed 20-person roster assumes no new entrances.','No learned re-identification or verified contact visibility.','Hidden floor positions use a labeled bounded zero-velocity forecast, not a measurement.','Relative mat calibration unchanged and unverified in physical units.','Sparse reference annotations are assistant-reviewed, not independently audited.']})
    print('Saved tracking experiment; selected validation variant',index,flush=True)

if __name__=='__main__':main()
