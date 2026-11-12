"""Second-round development search; first frozen test retained without rewriting."""
from frozen_holdout import *
from pipeline_eval import dense_motion
import trajectory_fit

def main():
    dest=OUT/'round2';dest.mkdir(exist_ok=True);old=read(OUT/'search/frozen_selection.json')['offline']['chosen'];folder=OUT/old['folder'];base=read(folder/'base_tracks.json');pred=read(folder/'predictions.json');data=read(OUT/'data/development_v1.json')['controlled'];labels=data['labels'];frames=data['val'];rows=[]
    for threshold in [.25,.4,.5,.6]:
        z,cf,ids=inputs(base,pred,threshold);before,_=original_arrays(base,ids)
        for cfg in trajectory_fit.configs():
            a,st,age=trajectory_fit.fit(z,cf,cfg);r={'threshold':threshold,'settings':cfg,'accuracy':identity_score(a,st,labels,frames,ids),'continuity':continuity(a,before),'dense_motion':dense_motion(a,ids)};rows.append(r)
        print('Trajectory threshold',threshold,flush=True)
    # New criteria are committed before running this search. Backup section is untouched.
    eligible=[r for r in rows if r['accuracy']['hidden_measured']<=2 and r['continuity']['candidate']['step_p99']<=1 and r['continuity']['candidate']['acceleration_p99']<=.8 and all(x['demeaned_rmse_pct']<1 for x in r['dense_motion']['rows'])]
    assert eligible,'No safe candidate; retain round1.'
    top=max(r['accuracy']['correct_by_tolerance_pct_width']['3'] for r in eligible);near=[r for r in eligible if r['accuracy']['correct_by_tolerance_pct_width']['3']>=top-1];chosen=min(near,key=lambda r:(-r['accuracy']['correct_by_tolerance_pct_width']['1'],r['continuity']['candidate']['acceleration_p99']))
    write(dest/'trajectory_search.json',{'scope':'Development-only selection after first reserved result inspected. First reserved scores remain historical. Backup5125..5265 remains untouched.','candidates':rows,'chosen':chosen,'base_pipeline':old});z,cf,ids=inputs(base,pred,chosen['threshold']);a,st,age=trajectory_fit.fit(z,cf,chosen['settings']);write(dest/'development_tracks.json',export(base,a,st,age,z,ids,chosen['settings']));print('CHOSEN',chosen['threshold'],chosen['settings'],chosen['accuracy']['correct_by_tolerance_pct_width'],chosen['continuity'],flush=True)
    # Diagnostic on previously seen first test. This is explicitly no longer fresh validation for round2.
    base=read(OUT/'reserved_evaluation/offline/base_tracks.json');pred=read(OUT/'reserved_evaluation/offline/predictions.json');z,cf,ids=inputs(base,pred,chosen['threshold']);a,st,age=trajectory_fit.fit(z,cf,chosen['settings']);tracks=export(base,a,st,age,z,ids,chosen['settings']);write(dest/'previous_test_tracks.json',tracks);scores=track_score(tracks,read(D/'labels.json'));before,_=original_arrays(read(LONG/'reference_tracks.json'),ids);write(dest/'previous_test_diagnostic.json',{'accuracy':scores,'continuity':continuity(a[380:],before[380:]),'scope':'Previously inspected reserved section, now diagnostic only for round2.'});print('PREVIOUS TEST',scores['all'],flush=True)

if __name__=='__main__':main()
