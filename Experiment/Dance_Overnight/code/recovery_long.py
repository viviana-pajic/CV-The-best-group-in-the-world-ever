"""Long-sequence recovery diagnostic; does not change the frozen recommended pipeline."""
from recovery_tracking import *
from detector_probe import merge
from frozen_holdout import identity_audit

def main():
    cv2.setNumThreads(2);torch.set_num_threads(4);out=OUT/'recovery/long';out.mkdir(parents=True,exist_ok=True);tracking=read(OUT/'detector/sequences/posture_frozen_960/tracking_selection.json');raw=read(OUT/'full_replay/raw_detections.json');saved=np.load(OUT/'reid_full/features.npz');feature_array=saved['features'];seed_array=saved['seeds'];saved.close();pred={};features={'seeds':seed_array,'frames':{}};offset=0
    for source in range(3250,5674):
        rows=merge(raw[str(source)],tracking['detector_merge_iou']);ff=feature_array[offset:offset+len(rows)];offset+=len(rows)
        if source>=4420:pred[str(source-4420)]=rows;features['frames'][str(source-4420)]=ff
    assert offset==len(feature_array);flowpath=out/'flow.npz'
    if flowpath.exists():flow=np.load(flowpath)['flow'].astype(np.float32)
    else:
        cap=cv2.VideoCapture(str(OUT/'full_replay/source.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,1170);alg=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);previous=None;flow=[]
        for k in range(1254):
            ok,im=cap.read();assert ok;gray=cv2.resize(cv2.cvtColor(im[280:600,100:1220],cv2.COLOR_BGR2GRAY),(560,160));flow.append(np.zeros((160,560,2),np.float32) if previous is None else alg.calc(previous,gray,None));previous=gray
        cap.release();flow=np.asarray(flow);np.savez_compressed(flowpath,flow=flow.astype(np.float16))
    checkpoint=torch.load(OUT/'reid/tuned_tail/best.pt',map_location='cpu',weights_only=True);prototypes=checkpoint['prototypes'].numpy();seeds=read(ROOT/'Dance_Corrected_Pilot/labels.json')['0'];base=read(OUT/'reid/tuned_tracking_results.json')['chosen'];results=[]
    for sim,margin in [(.7,.02),(.8,.05),(.9,.05)]:
        cfg={**base['config'],'recovery_similarity':sim,'recovery_margin':margin,'recovery_min_age':25,'recovery_confirmation':3,'recovery_detection_conf':.2};name=f'recovery_s{sim}_m{margin}';tracks=run_recovery(pred,seeds,features,cfg,flow,prototypes);write(out/f'{name}_tracks.json',tracks);r={'name':name,'config':cfg,'events':tracks['recovery_events'],'limited_identity':identity_audit(tracks),'coverage_by_block':{}}
        for first,last in [(4420,4719),(4720,4905),(4906,5265),(5266,5430),(5431,5673)]:
            rr=tracks['frames'][first-4420:last-4420+1];r['coverage_by_block'][f'{first}-{last}']=float(np.median([sum(o['status']=='observed' for o in row) for row in rr]))
        results.append(r);write(out/'results_partial.json',results);print(name,'events',len(r['events']),'identity',r['limited_identity']['matches_with_correct_initial_id'],r['limited_identity']['wrong_initial_id_matches'],'counts',r['coverage_by_block'],flush=True)
    write(out/'results.json',{'scope':'Diagnostic after earlier sections were inspected. The development clip triggered no recovery events, so these thresholds are not validated by that clip. Counts are not dancer recall or identity accuracy. No model/filter change and no claim of fresh heldout evaluation.','candidates':results})

if __name__=='__main__':main()
