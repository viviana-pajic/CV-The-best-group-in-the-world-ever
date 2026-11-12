"""Paired diagnostic of long-context identity tracking, not a new untouched test."""
import frozen_holdout as fh
from frozen_holdout import *
import trajectory_fit

def main():
    dest=OUT/'reid_full';cfg=read(OUT/'round2/frozen_selection.json')['selection'];results={'scope':'Diagnostic on previously inspected sections. Both pipelines use identical forward context4420..5673, detector, support checkpoint and trajectory settings. ReID model/weight selected on development only.','candidates':{}}
    for name,folder in [('color_motion',OUT/'full_replay'),('tuned_reid',dest)]:
        base=read(folder/'forward_tracks.json');pred=read(folder/'support/predictions.json')[1170:];z,cf,ids=inputs(base,pred,cfg['threshold']);a,st,age=trajectory_fit.fit(z,cf,cfg['settings']);tracks=export(base,a,st,age,z,ids,cfg['settings']);write(dest/f'{name}_forward_filtered.json',tracks);row={}
        for setname in ['holdout','backup']:
            fh.D=OUT/'data'/setname;row[setname]=fh.track_score(tracks,read(fh.D/'labels.json'))
        fh.D=OUT/'data/holdout';row['limited_identity']=fh.identity_audit(tracks);row['coverage_by_block']={}
        for first,last in [(4420,4719),(4720,4905),(4906,5265),(5266,5430),(5431,5673)]:
            rr=tracks['frames'][first-4420:last-4420+1];row['coverage_by_block'][f'{first}-{last}']={'median_observed':float(np.median([sum(o['status']=='observed' for o in r) for r in rr])),'median_model_points':float(np.median([sum(o['map_status']=='proxy' for o in r) for r in rr])),'note':'Counts are not identity accuracy or dancer recall.'}
        results['candidates'][name]=row;print('DIAGNOSTIC',name,{k:row[k]['all']['correct_by_tolerance_pct_width'] for k in ['holdout','backup']},{k:row['limited_identity'][k] for k in ['matched_observed_boxes','matches_with_correct_initial_id','wrong_initial_id_matches']},flush=True)
    write(dest/'diagnostic_results.json',results)

if __name__=='__main__':main()
