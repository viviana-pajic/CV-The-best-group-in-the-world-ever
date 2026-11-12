"""Same tracked boxes, new support head. All later evaluations are diagnostics."""
import frozen_holdout as fh
from frozen_holdout import *
import trajectory_fit
from pathlib import Path

def local_score(tracks,labels):
    records=[]
    for k,gt in labels.items():
        pp=[o for o in tracks['frames'][int(k)-4420] if o['status']=='observed' or o['status']=='predicted' and o['age_frames']<=25];links=match_boxes([o['box'] for o in gt],[o['box'] for o in pp],.5)
        for i,o in enumerate(gt):
            q=pp[links[i]] if i in links else None;state={'proxy':1,'estimated':2,'unknown':0}[q['map_status']] if q else 0;records.append(record(k,o,q['map'] if q else None,state,q is not None,q['id'] if q else None))
    return {'all':aggregate(records),'records':records}

def main():
    torch.set_num_threads(3);cv2.setNumThreads(2);dest=OUT/'ending_experiment/pipeline';dest.mkdir(exist_ok=True);selection=read(OUT/'ending_experiment/results.json')['chosen'];base=read(OUT/'full_replay/base_tracks.json');infer(selection['candidate'],base,OUT/'full_replay/source.avi',dest/'support','cuda');newpred=read(dest/'support/predictions.json');oldpred=read(OUT/'full_replay/support/predictions.json');frozen=read(OUT/'round2/frozen_selection.json')['selection'];forward=read(OUT/'full_replay/forward_tracks.json');results={'scope':'Later diagnostic, not another untouched test. Same frozen detector, IDs, source context and trajectory settings. Head and its development-selected threshold change. Ending training/validation are correlated; ending training frames are excluded from the numerical ending comparison.','selection':selection,'candidates':{}}
    for name,pred,threshold in [('previous_support',oldpred,frozen['threshold']),('expanded_support',newpred,selection['threshold'])]:
        z,cf,ids=inputs(base,pred,threshold);a,st,age=trajectory_fit.fit(z,cf,frozen['settings']);full=export(base,a,st,age,z,ids,frozen['settings']);write(dest/f'{name}_full_tracks.json',full)
        z,cf,ids=inputs(forward,pred[1170:],threshold);a,st,age=trajectory_fit.fit(z,cf,frozen['settings']);tracks=export(forward,a,st,age,z,ids,frozen['settings']);write(dest/f'{name}_forward_tracks.json',tracks);row={}
        for setname in ['holdout','backup']:
            fh.D=OUT/'data'/setname;row[setname]=fh.track_score(tracks,read(fh.D/'labels.json'))
        ending=read(OUT/'data/ending/labels.json');val=read(OUT/'data/ending/manifest.json')['val'];row['ending_development']=local_score(tracks,{str(k):ending[str(k)] for k in val});d=read(OUT/'data/development_v1.json')['controlled'];row['controlled_development']=identity_score(a,st,d['labels'],d['val'],ids);row['continuity']=continuity(a[380:],original_arrays(read(dest/'previous_support_forward_tracks.json'),ids)[0][380:]);results['candidates'][name]=row;write(dest/'results.json',results);print('ENDING PIPELINE',name,{k:row[k]['all']['correct_by_tolerance_pct_width'] for k in ['holdout','backup','ending_development']},flush=True)

if __name__=='__main__':main()
