"""Run trained heads on frozen detector tracks; development evaluation only."""
import sys,json,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
import cv2,numpy as np,torch
from support_train import SupportNet
from support_evaluate import predict,anchor,raw_copy,reportability
from ankle_pilot import maps,raw_metrics
from occlusion_pilot import evaluate
from map_kalman_pilot import frozen_hash,stationary,motion_stats

def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    def enc(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v))
    Path(p).write_text(json.dumps(obj,default=enc,indent=2))

def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    base=read(ROOT/'Dance_Tracking_Pilot/tracks.json')['roster'];dataset=read(OUT/'data/development_v1.json')['controlled'];labels=dataset['labels'];frames=dataset['val'];summary=[]
    for folder in sorted((OUT/'runs').glob('v1_*')):
        if not (folder/'complete.json').exists():continue
        cached=folder/'controlled_predictions.json'
        if not cached.exists():
            net=SupportNet(False).cuda().eval();ckpt=torch.load(folder/'best.pt',weights_only=True,map_location='cpu');net.load_state_dict(ckpt['state_dict']);cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));assoc=[];start=time.perf_counter()
            for k in range(300):
                ok,im=cap.read();assert ok;assoc.append(predict(net,im,[o for o in base['frames'][k] if o['status']=='observed']))
                if k%100==0:print(folder.name,'infer',k,flush=True)
            cap.release();write(cached,assoc);write(folder/'timing.json',{'seconds':time.perf_counter()-start,'frames':300});del net
        assoc=[{int(k):v for k,v in d.items()} for d in read(cached)]
        candidates=[]
        for threshold in [.1,.25,.4,.5,.6,.75,.9]:
            d=raw_copy(base,assoc,threshold);candidates.append({'threshold':threshold,'raw':raw_metrics(d,labels,frames),'visibility':reportability(d,labels,frames)})
        selected=max(candidates,key=lambda d:(d['raw']['joint_f1'],d['raw']['correct_within_3pct']));threshold=selected['threshold']
        after=maps(base,assoc,{'threshold':threshold},anchor);assert frozen_hash(base)==frozen_hash(after)
        st=stationary(base,after,labels);st['learned']=st.pop('kalman');m=motion_stats(base,after);m['metrics']['learned']=m['metrics'].pop('kalman')
        scores={'run':folder.name,'selection_scope':'Development only: former pilot test frames now used for training/validation. Separate 192–203.2s holdout remains untouched.','candidates':candidates,'selected':selected,'map':evaluate(after,labels,frames),'baseline':{'raw':raw_metrics(raw_copy(base,assoc,box=True),labels,frames),'map':evaluate(base,labels,frames)},'stationary':st,'motion':m}
        write(folder/'controlled_metrics.json',scores);write(folder/'controlled_tracks.json',after);summary.append(scores);write(OUT/'development_v1_results.json',summary)
        print(folder.name,'raw',selected['raw']['correct_within_3pct'],'/',selected['raw']['visible_references'],'map',scores['map']['floor_correct_within_3pct'],'threshold',threshold,flush=True)

if __name__=='__main__':main()
