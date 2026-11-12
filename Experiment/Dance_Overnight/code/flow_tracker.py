"""Adjacent-image motion ablation, without later manual tracking corrections."""
import sys,time,itertools,inspect
from pathlib import Path
import cv2,numpy as np
from tracker_lab import ROOT,OUT,read,write,features,associate,core
from tracking_runtime import run_roster

def prepare_flow():
    path=OUT/'tracking/dense_flow.npz'
    if path.exists():return np.load(path)['flow'].astype(np.float32)
    cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));flow=[];prev=None
    alg=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for k in range(300):
        ok,im=cap.read();assert ok
        gray=cv2.resize(cv2.cvtColor(im[280:600,100:1220],cv2.COLOR_BGR2GRAY),(560,160))
        ff=np.zeros((160,560,2),np.float32) if prev is None else alg.calc(prev,gray,None)
        flow.append(ff);prev=gray
        if k%100==0:print('Dense motion',k,flush=True)
    cap.release();flow=np.array(flow);np.savez_compressed(path,flow=flow.astype(np.float16));return flow

def displacement(field,box):
    x1,y1,x2,y2=np.asarray(box);w=x2-x1;h=y2-y1
    a=np.floor(((np.array([x1+.25*w,y1+.12*h])-np.array([100,280]))/2)).astype(int)
    b=np.ceil(((np.array([x1+.75*w,y1+.68*h])-np.array([100,280]))/2)).astype(int)
    a=np.maximum(a,0);b=np.minimum(b,[560,160]);patch=field[a[1]:b[1],a[0]:b[0]]
    if patch.size<20:return np.zeros(2)
    f=patch.reshape(-1,2)*2;m=np.median(f,axis=0);r=np.linalg.norm(f-m,axis=1);keep=r<max(1.5,np.percentile(r,60))
    return core.norm(np.median(f[keep],axis=0),12) if keep.any() else np.zeros(2)

def run(preds,seeds,feature,cfg,flow):
    prepared={k:[dict(o,feature=f) for o,f in zip(row,feature['frames'][k])] for k,row in preds.items()}
    sf={o['id']:np.asarray(f) for o,f in zip(seeds,feature['seeds'])}
    class FlowDancer(core.Dancer):
        def __init__(self,o):
            super().__init__(o);self.appearance=sf[o['id']].copy();self.initial_appearance=self.appearance.copy();self.flow_box=self.box.copy();self.flow_k=0
        def predicted(self,k):
            pb,pg=super().predicted(k);age=k-self.last
            if k>self.flow_k:
                self.flow_box+=np.tile(displacement(flow[k],self.flow_box),2);self.flow_k=k
            if age<=cfg['flow_max_gap']:
                alpha=cfg['flow_blend'];pb=(1-alpha)*pb+alpha*self.flow_box;pg=core.stage(core.bottom(pb))
            return pb,pg
        def update(self,b,k):
            super().update(b,k);self.flow_box=b.copy();self.flow_k=k
    return run_roster(prepared,seeds,cfg,make_track=FlowDancer,association=associate)

def main():
    cv2.setNumThreads(4);flow=prepare_flow();preds=read(ROOT/'Dance_Corrected_Pilot/predictions_detect.json');labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];ff=features(preds,seeds);base=read(OUT/'tracking/color_selection.json')['chosen'];results=[]
    for blend,gap,aw in itertools.product([.5,1.],[1,3,8],[.3,.5]):
        cfg={**base['config'],'flow_blend':blend,'flow_max_gap':gap,'appearance_weight':aw};name=f'flow_b{blend}_gap{gap}_a{aw}';data=run(preds,seeds,ff,cfg,flow);score=core.evaluate(data,labels,[165,175,185,195,285,299]);results.append({'name':name,'config':cfg,'validation':score});write(OUT/'tracking'/f'{name}_tracks.json',data);write(OUT/'tracking/flow_results.json',results)
        print(name,score['correct_initial_ids'],score['matched_boxes'],score['wrong_initial_ids'],flush=True)
    selected=max(results,key=lambda r:(r['validation']['correct_initial_ids'],r['validation']['matched_boxes']));write(OUT/'tracking/flow_selection.json',{'scope':'Development only; image motion computed from adjacent frames, no later annotation feedback.','chosen':selected,'color_baseline':base});print('CHOSEN',selected['name'],flush=True)

if __name__=='__main__':main()
