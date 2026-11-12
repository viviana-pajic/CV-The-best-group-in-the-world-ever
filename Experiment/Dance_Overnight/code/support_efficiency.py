"""Development-only cost of one, two and ten support-model crop views."""
from frozen_holdout import *
import trajectory_fit

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    while not (OUT/'detector_expanded/selected_support/development_score.json').exists():time.sleep(5)
    dest=OUT/'efficiency';dest.mkdir(exist_ok=True)
    selected=read(OUT/'ending_experiment/results.json')['chosen'];cfg=selected['candidate'];settings=read(OUT/'round2/frozen_selection.json')['selection']['settings'];base=read(OUT/'roi_repair/development_reference/base_tracks.json');d=read(OUT/'data/development_v1.json')['controlled'];results={}
    cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));images=[]
    for k in range(32):ok,im=cap.read();assert ok;images.append(im)
    cap.release()
    for mode in ['base','flip','crop5flip']:
        folder=dest/mode;modecfg={**cfg,'mode':mode}
        if mode=='crop5flip':pp=read(OUT/'roi_repair/development_reference/predictions.json')
        else:infer(modecfg,base,ROOT/'Dance_Controlled_Test/source.mp4',folder,'cuda');pp=read(folder/'predictions.json')
        z,cf,ids=inputs(base,pp,selected['threshold']);a,st,age=trajectory_fit.fit(z,cf,settings);score=identity_score(a,st,d['labels'],d['val'],ids)
        net,condition,_=load_model(OUT/'runs'/cfg['run']/cfg['checkpoint']);times=[]
        for k,im in enumerate(images):
            rows=[o for o in base['frames'][k] if o['status']=='observed'];torch.cuda.synchronize();start=time.perf_counter();predict_aug(net,im,rows,mode,condition);torch.cuda.synchronize()
            if k>=2:times.append(time.perf_counter()-start)
        del net;torch.cuda.empty_cache();results[mode]={'development':score,'warm_support_seconds_per_frame':float(np.median(times)),'warm_support_fps':float(1/np.median(times)),'timed_frames':30};write(dest/'results.json',{'scope':'Same controlled development images, detector boxes, IDs, V6 checkpoint, threshold0.25 and robust filter. Not a fresh test. Runtime includes crop preprocessing and support head only, excludes detection/tracking/rendering. Measured after detector training/evaluation queue finishes. No model selection from reserved sections.','candidates':results});print('EFFICIENCY',mode,results[mode],flush=True)

if __name__=='__main__':main()
