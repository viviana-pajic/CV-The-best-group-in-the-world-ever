"""Extend the frozen detector/trackers without resetting IDs or using holdout labels."""
import time,hashlib
from detector_probe import candidates,merge,YOLO,torch
from tracker_lab import ROOT,OUT,read,write,color_feature
from flow_tracker import run as run_flow
from tracking_runtime import run_roster
import cv2,numpy as np

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);out=OUT/'long_sequence';out.mkdir(exist_ok=True)
    first,last=4420,5080;n=last-first+1;start=time.perf_counter()
    source=out/'source.avi';writer=None;source_available=source.exists()
    if not source_available:
        writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),25,(1280,720));assert writer.isOpened()
    oldpred=read(ROOT/'Dance_Corrected_Pilot/predictions_detect.json');pred=read(out/'predictions.json') if (out/'predictions.json').exists() else dict(oldpred)
    oldff=read(OUT/'tracking/color_features.json');ff={'seeds':oldff['seeds'],'frames':dict(oldff['frames'])};seeds=read(ROOT/'Dance_Corrected_Pilot/labels.json')['0']
    model=YOLO(ROOT/'Dance_Corrected_Pilot/runs/detect/weights/best.pt');cap=cv2.VideoCapture(str(source if source_available else ROOT/'Dance_Controlled_Test/source.mp4'))
    motion=[];prev=None;alg=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for k in range(n):
        if source_available or k<300:ok,im=cap.read();assert ok
        else:im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{first+k:06}.png'));assert im is not None
        if writer:writer.write(im)
        if str(k) not in pred:pred[str(k)]=merge(candidates(model,im,960,'0'),.65)
        if k>=300:ff['frames'][str(k)]=[color_feature(im,o['box']).tolist() for o in pred[str(k)]]
        gray=cv2.resize(cv2.cvtColor(im[280:600,100:1220],cv2.COLOR_BGR2GRAY),(560,160))
        field=np.zeros((160,560,2),np.float32) if prev is None else alg.calc(prev,gray,None)
        motion.append(field);prev=gray
        if k%100==0:print('Long sequence preparation',k,flush=True);write(out/'predictions.json',pred)
    cap.release()
    if writer:writer.release()
    write(out/'predictions.json',pred);write(out/'color_features.json',ff);motion=np.array(motion);np.savez_compressed(out/'dense_flow.npz',flow=motion.astype(np.float16))
    cfg=read(OUT/'tracking/flow_selection.json')['chosen']['config'];flow=run_flow(pred,seeds,ff,cfg,motion);write(out/'flow_tracks.json',flow)
    reference=run_roster(pred,seeds,read(ROOT/'Dance_Tracking_Pilot/settings.json'));write(out/'reference_tracks.json',reference)
    old=read(OUT/'tracking/flow_b1.0_gap1_a0.5_tracks.json')
    assert all(a['id']==b['id'] and a['status']==b['status'] and np.allclose(a['box'],b['box']) for aa,bb in zip(flow['frames'][:300],old['frames']) for a,b in zip(aa,bb)), 'Extension must preserve development tracking'
    check=cv2.VideoCapture(str(source));decoded=0
    while True:
        ok,im=check.read()
        if not ok:break
        decoded+=1
    check.release();assert decoded==n
    write(out/'provenance.json',{'source_first':first,'source_last':last,'frames':n,'holdout_local_range':[380,660],'initialization':'Same20 visually reviewed seeds at4420; no subsequent manual reseeding or heldout label feedback.','detector':'Previous corrected YOLO26m,960px crops, mergeIoU.65; original300-frame detection cache preserved.','video':'Lossless FFV1; first300 decoded controlled pilot, remaining source PNGs.','seconds_including_detection_flow_and_tracking':time.perf_counter()-start,'timing_limit':'Concurrent GPU training; not a standalone speed benchmark.','verified_decoded_frames':decoded})
    print('Long sequence ready',n,'frames',flush=True)

if __name__=='__main__':main()
