"""Apply precommitted development rules; never read confirmation images or labels."""
from frozen_holdout import *
import trajectory_fit

def main():
    dest=OUT/'final_confirmation';path=dest/'frozen_selection.json'
    if path.exists():print('Already frozen',flush=True);return
    while not (OUT/'detector_expanded/selected_support/development_score.json').exists():time.sleep(5)
    protocol=read(dest/'selection_protocol.json');expanded=read(OUT/'detector_expanded/tracking_selection.json')['chosen'];newscore=read(OUT/'detector_expanded/selected_support/development_score.json');oldcfg=read(OUT/'round2/frozen_selection.json')['selection'];base=read(OUT/'full_replay/forward_tracks.json')
    for key in ['frames','diagnostics']:base[key]=base[key][:300]
    pp=read(OUT/'full_replay/support/predictions.json')[1170:1470];z,cf,ids=inputs(base,pp,oldcfg['threshold']);a,st,age=trajectory_fit.fit(z,cf,oldcfg['settings']);d=read(OUT/'data/development_v1.json')['controlled'];oldscore=identity_score(a,st,d['labels'],d['val'],ids);v=expanded['validation'];accept=v['correct_initial_ids']>=100 and v['wrong_initial_ids']<=1 and newscore['correct_by_tolerance_pct_width']['3']>=oldscore['correct_by_tolerance_pct_width']['3']
    detector=({'weights':f"detector_expanded/runs/{expanded['model']}/weights/best.pt",'config':expanded['config'],'detector_merge_iou':expanded['detector_merge_iou'],'source':'expanded development-selected detector'} if accept else {'weights':'detector/runs/posture_frozen/weights/best.pt',**read(OUT/'detector/sequences/posture_frozen_960/tracking_selection.json'),'source':'original frozen detector retained'})
    freeze={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_protocol_sha256':hashlib.sha256((dest/'selection_protocol.json').read_bytes()).hexdigest(),'support':protocol['support']['candidate'],'threshold':protocol['support']['threshold'],'settings':protocol['filter'],'detector':detector,'expanded_detector_accepted':accept,'detector_development':{'original_same_head':oldscore,'expanded_same_head':newscore,'expanded_identity':v},'confirmation':'No confirmation image labels or model outputs inspected to make this choice. Three variants fixed in selection_protocol.json.','sha256':{'support':protocol['support_checkpoint_sha256'],'detector':hashlib.sha256((OUT/detector['weights']).read_bytes()).hexdigest()}}
    write(path,freeze);print('LATE CHOICE FROZEN; expanded detector accepted:',accept,flush=True);print('Original and expanded floor scores:',oldscore['correct_by_tolerance_pct_width'],newscore['correct_by_tolerance_pct_width'],flush=True)

if __name__=='__main__':main()
