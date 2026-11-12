"""Diagnostic replay of the development-selected, choreography-tuned ReID tracker."""
from reid_trial import *
from pipeline_infer import infer
from filter_lab import inputs,export
import trajectory_fit

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);out=OUT/'reid_full';out.mkdir(exist_ok=True);selection=read(OUT/'reid/tuned_tracking_results.json')['chosen'];assert selection['name']=='tuned_tail_seed_a0.3';checkpoint=torch.load(OUT/'reid/tuned_tail/best.pt',map_location='cpu',weights_only=True);source=OUT/'full_replay/source.avi';raw=read(OUT/'full_replay/raw_detections.json');tracking=read(OUT/'detector/sequences/posture_frozen_960/tracking_selection.json');pred={k:merge(raw[str(k)],tracking['detector_merge_iou']) for k in range(3250,5674)};seeds=read(ROOT/'Dance_Corrected_Pilot/labels.json')['0'];cache=out/'features.npz';features={};gray=[];cap=cv2.VideoCapture(str(source));net=None if cache.exists() else model()
    if net:net.load_state_dict(checkpoint['state_dict'])
    saved=np.load(cache) if cache.exists() else None;cached_features=saved['features'] if saved is not None else None;offset=0
    for k in range(3250,5674):
        ok,im=cap.read();assert ok;gray.append(cv2.resize(cv2.cvtColor(im[280:600,100:1220],cv2.COLOR_BGR2GRAY),(560,160)));n=len(pred[k])
        if saved is not None:features[k]=cached_features[offset:offset+n];offset+=n
        else:features[k]=np.asarray(extract(net,im,pred[k]),np.float32).reshape(-1,512)
        if k==4420:seedfeatures=saved['seeds'] if saved is not None else np.asarray(extract(net,im,seeds),np.float32)
        if (k-3250)%300==0:print('Full tuned ReID features',k,flush=True)
    cap.release()
    if saved is None:np.savez_compressed(cache,features=np.concatenate(list(features.values())),seeds=seedfeatures,counts=np.array([len(pred[k]) for k in pred]))
    del net;torch.cuda.empty_cache();parts={}
    for direction,order in [('forward',list(range(4420,5674))),('backward',list(range(4420,3249,-1)))]:
        path=out/f'{direction}_tracks.json'
        if path.exists():parts[direction]=read(path);continue
        flow=[];prev=None;alg=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        for k in order:
            im=gray[k-3250];flow.append(np.zeros((160,560,2),np.float32) if prev is None else alg.calc(prev,im,None));prev=im
        pp={str(i):pred[k] for i,k in enumerate(order)};ff={'seeds':seedfeatures,'frames':{str(i):features[k] for i,k in enumerate(order)}};parts[direction]=run_flow(pp,seeds,ff,selection['config'],np.array(flow));write(path,parts[direction]);del flow;print('Tuned ReID tracking',direction,flush=True)
    base=copy.deepcopy(parts['forward'])
    for key in ['frames','diagnostics']:base[key]=parts['backward'][key][:0:-1]+parts['forward'][key]
    write(out/'base_tracks.json',base);cfg=read(OUT/'round2/frozen_selection.json');infer(cfg['base_pipeline']['support'],base,source,out/'support','cuda',reuse=OUT/'full_replay/support');pp=read(out/'support/predictions.json');z,cf,ids=inputs(base,pp,cfg['selection']['threshold']);a,st,age=trajectory_fit.fit(z,cf,cfg['selection']['settings']);write(out/'tracks.json',export(base,a,st,age,z,ids,cfg['selection']['settings']));write(out/'provenance.json',{'selection':selection,'scope':'Later diagnostic only. Previous reserved sets have already been inspected. Same detector/support/filter; new identity embedding trained only on14 controlled training frames, selected on6 development frames. One initialization, forward/backward tracking. No new claim of untouched evaluation.'});print('Tuned ReID full replay complete',flush=True)

if __name__=='__main__':
    from detector_probe import merge
    main()
