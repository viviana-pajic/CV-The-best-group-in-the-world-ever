"""Variable-length roster runtime, preserving the earlier pilot association/map logic.
Copied from Dance_Floor_Contact/occlusion_pilot.py; factory arguments avoid mutable globals.
"""
import numpy as np
from occlusion_pilot import Dancer,associate,inside,norm
from compare import stage,bottom

def run_roster(preds,seeds,config,make_track=Dancer,association=associate):
    tracks=[make_track(o) for o in seeds];rows=[];diagnostics=[]
    for k in range(len(preds)):
        det=[o for o in preds[str(k)] if o['score']>=config['low_conf']]
        links,unused=association(tracks,det,k,config) if k else ({},set())
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
