"""Comparable full-context diagnostic scores across previously inspected sections."""
import frozen_holdout as fh
from frozen_holdout import *
from evaluate_final_confirmation import score

if __name__=='__main__':
    prov=read(OUT/'late_full_replay/provenance.json');variants={'incumbent':read(OUT/'ending_experiment/pipeline/previous_support_full_tracks.json'),'expanded_head_only':read(OUT/'ending_experiment/pipeline/expanded_support_full_tracks.json'),'expanded_detector_and_head':read(OUT/prov['tracks'])};results={'scope':'Later diagnostics on previously inspected sections; not additional fresh tests. All variants retain full2424-frame map-fitting context. The head-only row keeps the original detector and IDs. The full candidate also changes the detector/association. Ending development has only8 visible references in2 correlated frames.','candidates':{}}
    for name,tracks in variants.items():
        row={}
        for split in ['holdout','backup']:
            fh.D=OUT/'data'/split;row[split]=score(tracks,read(fh.D/'labels.json'))
        ending=read(OUT/'data/ending/labels.json');frames=read(OUT/'data/ending/manifest.json')['val'];rr=[]
        for k in frames:
            gt=ending[str(k)];pp=[o for o in tracks['frames'][k-3250] if o['status']=='observed' or o['status']=='predicted' and o['age_frames']<=25];links=match_boxes([o['box'] for o in gt],[o['box'] for o in pp],.5)
            for i,o in enumerate(gt):
                q=pp[links[i]] if i in links else None;state={'proxy':1,'estimated':2,'unknown':0}[q['map_status']] if q else 0;rr.append(record(k,o,q['map'] if q else None,state,q is not None,q['id'] if q else None))
        row['ending_development']={'all':aggregate(rr),'records':rr};row['coverage']={}
        for section,(lo,hi) in {'clearer_development':(4420,4719),'crowd':(4906,5265),'ending':(5315,5673)}.items():
            rows=tracks['frames'][lo-3250:hi-3250+1];row['coverage'][section]={'median_observed_tracks':float(np.median([sum(o['status']=='observed' for o in rr) for rr in rows])),'median_measured_map_points':float(np.median([sum(o['map_status']=='proxy' for o in rr) for rr in rows])),'median_estimated_map_points':float(np.median([sum(o['map_status']=='estimated' for o in rr) for rr in rows]))}
        results['candidates'][name]=row;print(name,{k:row[k]['all']['correct_by_tolerance_pct_width'] for k in ['holdout','backup','ending_development']},flush=True)
    write(OUT/'diagnostics/latest_pipeline_sections.json',results)
