"""Qualitative whole-extracted-section stress test with one middle-frame initialization."""
import copy,time
from frozen_holdout import *
from tracking_runtime import run_roster
from pipeline_infer import infer
import trajectory_fit

DEST=OUT/'full_replay';FIRST=3250;SEED=4420;LAST=5673

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);DEST.mkdir(exist_ok=True);assert (OUT/'round2/frozen_selection.json').exists();freeze=read(OUT/'round2/frozen_selection.json');cfg=freeze['base_pipeline'];tracking=read(OUT/'detector/sequences/posture_frozen_960/tracking_selection.json');seeds=read(ROOT/'Dance_Corrected_Pilot/labels.json')['0'];start=time.perf_counter();path=DEST/'source.avi'
    if not path.exists():
        w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),25,(1280,720));assert w.isOpened();cap=cv2.VideoCapture(str(LONG/'source.avi'))
        for k in range(FIRST,LAST+1):
            if SEED<=k<=5080:ok,im=cap.read();assert ok
            else:im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{k:06}.png'));assert im is not None
            w.write(im)
        w.release();cap.release()
    rawpath=DEST/'raw_detections.json';raw=read(rawpath) if rawpath.exists() else {str(int(k)+SEED):v for k,v in read(LONG/'frozen/raw.json').items()};model=YOLO(OUT/'detector/runs/posture_frozen/weights/best.pt');cap=cv2.VideoCapture(str(path));pred={};features={};images=[]
    for k in range(FIRST,LAST+1):
        ok,im=cap.read();assert ok
        if str(k) not in raw:raw[str(k)]=candidates(model,im,960,'0')
        pp=merge(raw[str(k)],tracking['detector_merge_iou']);pred[k]=pp;features[k]=[color_feature(im,o['box']).tolist() for o in pp];images.append(cv2.resize(cv2.cvtColor(im[280:600,100:1220],cv2.COLOR_BGR2GRAY),(560,160)))
        if k==SEED:seedfeatures=[color_feature(im,o['box']).tolist() for o in seeds]
        if (k-FIRST)%200==0:write(rawpath,raw);print('Full detector',k,flush=True)
    cap.release();write(rawpath,raw);del model;torch.cuda.empty_cache();direction_tracks={};direction_baselines={}
    for direction,order in [('forward',list(range(SEED,LAST+1))),('backward',list(range(SEED,FIRST-1,-1)))]:
        cache=DEST/f'{direction}_tracks.json'
        if cache.exists():direction_tracks[direction]=read(cache);direction_baselines[direction]=read(DEST/f'{direction}_box_baseline.json');continue
        alg=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);motion=[];previous=None
        for k in order:
            gray=images[k-FIRST];motion.append(np.zeros((160,560,2),np.float32) if previous is None else alg.calc(previous,gray,None));previous=gray
        pp={str(i):pred[k] for i,k in enumerate(order)};ff={'seeds':seedfeatures,'frames':{str(i):features[k] for i,k in enumerate(order)}};tracks=run_flow(pp,seeds,ff,tracking['config'],np.array(motion));direction_tracks[direction]=tracks;write(cache,tracks)
        baseline=run_roster(pp,seeds,read(ROOT/'Dance_Tracking_Pilot/settings.json'));direction_baselines[direction]=baseline;write(DEST/f'{direction}_box_baseline.json',baseline);del motion;print('Full tracking',direction,flush=True)
    def join(parts):
        result=copy.deepcopy(parts['forward'])
        for key in ['frames','diagnostics']:result[key]=parts['backward'][key][:0:-1]+parts['forward'][key]
        return result
    base=join(direction_tracks);baseline=join(direction_baselines);write(DEST/'base_tracks.json',base);write(DEST/'same_detector_box_baseline.json',baseline)
    infer(cfg['support'],base,path,DEST/'support','cuda');pred=read(DEST/'support/predictions.json');selection=freeze['selection'];z,cf,ids=inputs(base,pred,selection['threshold']);a,st,age=trajectory_fit.fit(z,cf,selection['settings']);tracks=export(base,a,st,age,z,ids,selection['settings']);write(DEST/'tracks.json',tracks)
    write(DEST/'provenance.json',{'first':FIRST,'last':LAST,'frames':LAST-FIRST+1,'seconds':(LAST-FIRST+1)/25,'initialization':'Twenty reviewed boxes at source4420 only. Track forward and backward from that frame, then join chronologically. Backward tracking and final map fit use future information. No other manual reseeds.','evaluation':'Qualitative stress test; includes training and validation frames. Not independent accuracy evidence. Unseen backup images were processed after both pipeline variants were frozen; their labels were not used.','baseline':'Same newly trained detector with previous box map/association logic. This is not the old detector baseline used in first reserved comparison.','runtime_seconds':time.perf_counter()-start})
    print('Full replay complete',len(tracks['frames']),flush=True)

if __name__=='__main__':main()
