"""Post-confirmation diagnostic: bounded weak-contact estimates, never measurements."""
from frozen_holdout import *
import frozen_holdout as fh
import trajectory_fit
from evaluate_final_confirmation import score

def main():
    torch.set_num_threads(2);dest=OUT/'weak_contact';dest.mkdir(exist_ok=True);cfg=read(OUT/'final_confirmation/frozen_selection.json');base=read(OUT/'late_full_replay/base_tracks.json');pred=read(OUT/'late_full_replay/support/predictions.json');original=read(OUT/'late_full_replay/tracks.json')
    protocol={'scope':'Exploratory diagnostic motivated by inspection of the last confirmation. That block is no longer a fresh test for this change. Fixed settings, no grid search. No detector/ID changes. Never replaces an existing map point. New points are always hollow estimates, not visible contacts. Hidden coordinate accuracy is unknown.','augmentation_rms_limit_pixels':3.,'maximum_anchor_distance_frames':50,'anchor_gate_pct_width':3.,'anchor_gate_speed_pct_width_per_frame':.3,'weak_observation_confidence':.1}
    write(dest/'protocol.json',protocol);z,confidence,ids=inputs(base,pred,cfg['threshold']);strong=z.copy();filled=copy.deepcopy(original);counts={'weak_observations':0,'new_estimated_track_frames':0};idx={v:i for i,v in enumerate(ids)}
    for k,row in enumerate(base['frames']):
        for o in row:
            j=idx[o['id']];p=pred[k].get(str(o['id']))
            if o['status']!='observed' or not p or np.isfinite(z[k,j]).all():continue
            if p.get('augmentation_rms_pixels',1e9)>protocol['augmentation_rms_limit_pixels']:continue
            point=stage(p['point'])
            if not (0<=point[0]<=100 and 0<=point[1]<=62.5):continue
            z[k,j]=point;confidence[k,j]=protocol['weak_observation_confidence'];counts['weak_observations']+=1
    a,st,age=trajectory_fit.fit(z,confidence,cfg['settings']);trusted={ident:np.array([k for k,row in enumerate(original['frames']) if any(o['id']==ident and o['map_status']=='proxy' for o in row)]) for ident in ids};original_index=[{o['id']:o for o in row} for row in original['frames']]
    for k,row in enumerate(filled['frames']):
        for o in row:
            j=idx[o['id']]
            if o['map_status']!='unknown' or o['status']!='observed' or not np.isfinite(a[k,j]).all():continue
            anchors=trusted[o['id']]
            if not len(anchors):continue
            near=int(anchors[np.argmin(abs(anchors-k))]);dt=abs(near-k)
            if dt>protocol['maximum_anchor_distance_frames']:continue
            point=original_index[near][o['id']]['map']
            if np.linalg.norm(a[k,j]-point)>protocol['anchor_gate_pct_width']+protocol['anchor_gate_speed_pct_width_per_frame']*dt:continue
            o.update(map=a[k,j].tolist(),map_status='estimated',map_age_frames=dt,uncertainty=1.+.1*dt,map_estimate_reason='Weak image support plus offline trajectory; near trusted anchor. Contact visibility is not asserted.');counts['new_estimated_track_frames']+=1
    write(dest/'tracks.json',filled);data=read(OUT/'data/development_v1.json')['controlled'];controlled={}
    for name,tracks in [('original',original),('weak_estimates',filled)]:
        part={**tracks,'frames':tracks['frames'][1170:1470]};aa,ss=original_arrays(part,ids);controlled[name]=identity_score(aa,ss,data['labels'],data['val'],ids)
    fh.D=OUT/'data/final_confirmation';labels=read(fh.D/'labels.json');diagnostic={name:score(tracks,labels) for name,tracks in [('original',original),('weak_estimates',filled)]}
    write(dest/'results.json',{'protocol':protocol,'counts':counts,'controlled_development':controlled,'previous_confirmation_diagnostic':diagnostic});print(counts,flush=True)
    for name,x in diagnostic.items():print(name,x['all']['correct_by_tolerance_pct_width'],'available',x['all']['available_visible'],'hidden_estimates',x['all']['hidden_estimated'],flush=True)

if __name__=='__main__':main()
