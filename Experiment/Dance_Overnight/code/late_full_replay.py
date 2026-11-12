"""Replay a development-approved detector without changing any frozen setting."""
from frozen_holdout import *
import trajectory_fit

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    while not (OUT/'final_confirmation/frozen_selection.json').exists():time.sleep(5)
    frozen=read(OUT/'final_confirmation/frozen_selection.json');dest=OUT/'late_full_replay';dest.mkdir(exist_ok=True);source=OUT/'full_replay/source.avi';first,seed,last=3250,4420,5673
    while not (OUT/'efficiency/results.json').exists() or len(read(OUT/'efficiency/results.json')['candidates'])<3:time.sleep(5)
    if not frozen['expanded_detector_accepted']:
        write(dest/'provenance.json',{'detector_changed':False,'base_tracks':'full_replay/base_tracks.json','support_predictions':'ending_experiment/pipeline/support/predictions.json','tracks':'ending_experiment/pipeline/expanded_support_full_tracks.json','selection':'final_confirmation/frozen_selection.json'});print('Reusing exact original-detector expanded-support replay.',flush=True);return
    cfg=frozen['detector'];seeds=read(ROOT/'Dance_Corrected_Pilot/labels.json')['0'];rawpath=dest/'raw_detections.json';raw=read(rawpath) if rawpath.exists() else {};pred={};features={};images=[];model=YOLO(OUT/cfg['weights']);cap=cv2.VideoCapture(str(source))
    for k in range(first,last+1):
        ok,im=cap.read();assert ok
        if str(k) not in raw:raw[str(k)]=candidates(model,im,960,'0')
        pp=merge(raw[str(k)],cfg['detector_merge_iou']);pred[k]=pp;features[k]=[color_feature(im,o['box']).tolist() for o in pp];images.append(cv2.resize(cv2.cvtColor(im[280:600,100:1220],cv2.COLOR_BGR2GRAY),(560,160)))
        if k==seed:seedfeatures=[color_feature(im,o['box']).tolist() for o in seeds]
        if (k-first)%300==0:write(rawpath,raw);print('Late detector',k,flush=True)
    cap.release();write(rawpath,raw);del model;torch.cuda.empty_cache();parts={}
    for direction,order in [('forward',list(range(seed,last+1))),('backward',list(range(seed,first-1,-1)))]:
        path=dest/f'{direction}_tracks.json'
        if path.exists():parts[direction]=read(path);continue
        alg=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);motion=[];previous=None
        for k in order:
            gray=images[k-first];motion.append(np.zeros((160,560,2),np.float32) if previous is None else alg.calc(previous,gray,None));previous=gray
        pp={str(i):pred[k] for i,k in enumerate(order)};ff={'seeds':seedfeatures,'frames':{str(i):features[k] for i,k in enumerate(order)}};parts[direction]=run_flow(pp,seeds,ff,cfg['config'],np.array(motion));write(path,parts[direction]);del motion;print('Late tracking',direction,flush=True)
    base=copy.deepcopy(parts['forward'])
    for key in ['frames','diagnostics']:base[key]=parts['backward'][key][:0:-1]+parts['forward'][key]
    write(dest/'base_tracks.json',base);infer(frozen['support'],base,source,dest/'support','cuda');pp=read(dest/'support/predictions.json');z,cf,ids=inputs(base,pp,frozen['threshold']);a,st,age=trajectory_fit.fit(z,cf,frozen['settings']);tracks=export(base,a,st,age,z,ids,frozen['settings']);write(dest/'tracks.json',tracks);write(dest/'provenance.json',{'detector_changed':True,'base_tracks':'late_full_replay/base_tracks.json','support_predictions':'late_full_replay/support/predictions.json','tracks':'late_full_replay/tracks.json','selection':'final_confirmation/frozen_selection.json','context':'Full2424frames; twenty reviewed seeds at4420, forward and backward tracking. Offline trajectories use future frames. Includes training frames; quantitative confirmation uses only precommitted new labels.'});print('Late replay complete',flush=True)

if __name__=='__main__':main()
