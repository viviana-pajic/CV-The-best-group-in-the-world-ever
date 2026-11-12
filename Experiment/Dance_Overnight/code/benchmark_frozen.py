"""Warm runtime measurements, after training/evaluation GPU queues have finished."""
import time,platform
from frozen_holdout import *

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);chosen=read(OUT/'search/frozen_selection.json')['offline']['chosen'];cfg=chosen['support'];members=cfg.get('members',[cfg]);nets=[]
    for m in members:
        net,mask,_=load_model(OUT/'runs'/m['run']/m['checkpoint']);nets.append((net,mask,m))
    weights=OUT/'detector/runs/posture_frozen/weights/best.pt' if chosen['tracker']=='frozen' else ROOT/'Dance_Corrected_Pilot/runs/detect/weights/best.pt';detector=YOLO(weights)
    base=read(OUT/chosen['folder']/'base_tracks.json');cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));images=[]
    for k in range(32):ok,im=cap.read();assert ok;images.append(im)
    cap.release();times=[];counts=[]
    for k,im in enumerate(images):
        torch.cuda.synchronize();start=time.perf_counter();dd=candidates(detector,im,960,'0');torch.cuda.synchronize();dt=time.perf_counter()-start
        rows=[o for o in base['frames'][k] if o['status']=='observed'];start=time.perf_counter()
        pp=[]
        for net,mask,m in nets:
            p=predict_aug(net,im,rows,m['mode'],mask)
            if m['decoder']!='point':
                for q in p.values():q['point']=q[m['decoder']]
            pp.append(p)
        if len(pp)==2:ensemble(*pp,cfg['ensemble_mode'])
        torch.cuda.synchronize();pt=time.perf_counter()-start
        if k>=2:times.append([dt,pt]);counts.append(len(rows))
    arr=np.array(times);folder=OUT/'reserved_evaluation/offline';tracks=read(folder/'base_tracks.json');pred=read(folder/'predictions.json');start=time.perf_counter();z,cf,ids=inputs(tracks,pred,chosen['threshold']);filtered(z,cf,chosen['settings']);post=time.perf_counter()-start
    data={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'frames_timed':len(times),'warmup_frames':2,'median_detected_rois_used_by_head':float(np.median(counts)),'detector_seconds_per_frame_median':float(np.median(arr[:,0])),'support_seconds_per_frame_median':float(np.median(arr[:,1])),'combined_seconds_per_frame_median':float(np.median(arr.sum(1))),'combined_core_inference_fps':float(1/np.median(arr.sum(1))),'offline_filter_seconds_for_661_frames':post,'scope':'Warm sequential detector and support-head inference on30 development frames, preloaded images. Includes support crop preparation. Excludes video decoding, tracking/appearance/flow, drawing, encoding, model loading and disk writes. Therefore core inference FPS is not end-to-end application FPS. No concurrent training intended.'}
    write(OUT/'diagnostics/runtime.json',data);print(data,flush=True)

if __name__=='__main__':main()
