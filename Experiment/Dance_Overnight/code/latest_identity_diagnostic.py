"""Limited earlier identity audit for the latest candidate; diagnostic, not a new test."""
import frozen_holdout as fh
from frozen_holdout import *

if __name__=='__main__':
    fh.D=OUT/'data/holdout';provenance=read(OUT/'late_full_replay/provenance.json');new=read(OUT/provenance['tracks']);old=read(OUT/'ending_experiment/pipeline/previous_support_full_tracks.json');results={'scope':'Later diagnostic using already inspected limited identity references. Four held-pose frames/80 bodies with original seed IDs. Does not audit later crowded motion or the new confirmation block. Full2424-frame map-fit context is retained, then arrays are indexed from4420 for scoring.','candidates':{}}
    for name,tracks in [('incumbent',old),('expanded',new)]:
        for key in ['frames','diagnostics']:tracks[key]=tracks[key][1170:]
        result=fh.identity_audit(tracks);results['candidates'][name]=result;print(name,{k:result[k] for k in ['box_references','matched_observed_boxes','matches_with_correct_initial_id','wrong_initial_id_matches']},flush=True)
    write(OUT/'diagnostics/latest_identity_diagnostic.json',results)
