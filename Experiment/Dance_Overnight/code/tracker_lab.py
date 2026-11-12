"""Fixed detector, appearance-assisted roster association. Development tuning only."""
import sys,copy,itertools,time,argparse
from pathlib import Path
import numpy as np,cv2
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
import occlusion_pilot as core
from compare import stage,bottom,overlap
from evaluate_support import read,write

def color_feature(im,box):
    x1,y1,x2,y2=map(round,box);x1=max(0,x1);x2=min(1280,x2);y1=max(0,y1);y2=min(720,y2);patch=im[y1:y2,x1:x2]
    if not patch.size:return np.zeros(384)
    hsv=cv2.cvtColor(cv2.resize(patch,(40,80)),cv2.COLOR_BGR2HSV);features=[]
    for a,b in [(5,45),(35,76)]:
        region=hsv[a:b,5:35];mask=np.uint8(region[:,:,2]>35)*255;hist=cv2.calcHist([region],[0,1,2],mask,[12,4,4],[0,180,0,256,0,256]).flatten();hist/=max(hist.sum(),1);features.append(np.sqrt(hist))
    f=np.concatenate(features);return f/max(np.linalg.norm(f),1e-9)

def features(preds,seeds):
    out=OUT/'tracking';out.mkdir(exist_ok=True);path=out/'color_features.json'
    if path.exists():return read(path)
    cache={'seeds':[],'frames':{}};cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'))
    for k in range(300):
        ok,im=cap.read();assert ok
        cache['frames'][str(k)]=[color_feature(im,o['box']).tolist() for o in preds[str(k)]]
        if not k:cache['seeds']=[color_feature(im,o['box']).tolist() for o in seeds]
    cap.release();write(path,cache);return cache

def associate(tracks,det,k,cfg):
    matches={};unused=set(range(len(det)))
    for threshold in [cfg['high_conf'],cfg['low_conf']]:
        ids=[i for i,t in enumerate(tracks) if i not in matches and k-t.last<=cfg['identity_memory_frames']];candidates=[j for j in sorted(unused) if det[j]['score']>=threshold]
        if not ids or not candidates:continue
        cost=np.full((len(ids),len(candidates)+len(ids)),cfg.get('unmatched_cost',.96))
        for ii,i in enumerate(ids):
            t=tracks[i];pb,pg=t.predicted(k);age=k-t.last;h=max(45,pb[3]-pb[1]);w=max(30,pb[2]-pb[0]);gate=cfg['ground_gate']+min(age,25)*.1
            for jj,j in enumerate(candidates):
                b=np.array(det[j]['box']);g=stage(bottom(b));gd=np.linalg.norm(g-pg);dc=core.center(b)-core.center(pb);cd=np.linalg.norm(dc/[max(w,60),max(h,90)]);iou=overlap(pb,b)
                # Reject a sudden loss of the lower body, but allow genuine crouching
                # when the bottom stays near the previous floor level.
                if cfg.get('lower_body_ratio') and age<=cfg.get('extent_gate_frames',3):
                    minimum=max(.4,cfg['lower_body_ratio']-.05*(age-1))
                    if (b[3]-b[1])<minimum*h and b[3]<pb[3]-cfg.get('bottom_rise_fraction',.15)*h:
                        cost[ii,jj]=99;continue
                    if cfg.get('growth_ratio') and (b[3]-b[1])>cfg['growth_ratio']*h and b[3]>pb[3]+.3*h:
                        cost[ii,jj]=99;continue
                rescue=age<=10 and iou>.30 and cd<.65
                if (gd>gate and not rescue and cfg['ground_weight']>0) or cd>cfg.get('center_gate',1.3) or abs(dc[0])>110:cost[ii,jj]=99;continue
                gw=cfg['ground_weight'];base=gw*min(gd/5,2)+.25*(1-iou)+(1-gw-.25)*min(cd,2)
                if rescue:base=min(base,.55*(1-iou)+.45*cd)
                f=np.asarray(det[j]['feature']);distance=1-max(float(t.appearance@f),float(t.initial_appearance@f) if cfg.get('initial_gallery') else -1)
                ac=min(max(distance,0)/cfg.get('appearance_distance_scale',.35),2);aw=cfg['appearance_weight'];c=(1-aw)*base+aw*ac+.06*(1-det[j]['score'])
                if age>75 and (det[j]['score']<.35 or c>.60 or (aw and distance>.55)):c=99
                if threshold==cfg['low_conf'] and (c>.70 or age>10):c=99
                cost[ii,jj]=c
        rr,cc=linear_sum_assignment(cost)
        for ii,jj in zip(rr,cc):
            if jj<len(candidates) and cost[ii,jj]<cfg.get('unmatched_cost',.96):
                i,j=ids[ii],candidates[jj];matches[i]=j;unused.remove(j)
                if det[j]['score']>=cfg['high_conf'] and cost[ii,jj]<.65:
                    f=np.asarray(det[j]['feature']);tracks[i].appearance=.9*tracks[i].appearance+.1*f;tracks[i].appearance/=max(np.linalg.norm(tracks[i].appearance),1e-9)
    return matches,unused

def run(preds,seeds,feature,cfg):
    prepared={k:[dict(o,feature=f) for o,f in zip(row,feature['frames'][k])] for k,row in preds.items()}
    seedfeat={o['id']:np.asarray(f) for o,f in zip(seeds,feature['seeds'])};original=core.Dancer;original_associate=core.associate
    class FeatureDancer(original):
        def __init__(self,o):super().__init__(o);self.appearance=seedfeat[o['id']].copy();self.initial_appearance=self.appearance.copy()
    try:
        core.Dancer=FeatureDancer;core.associate=associate;return core.run_roster(prepared,seeds,cfg)
    finally:core.Dancer=original;core.associate=original_associate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--feature',choices=['color','deep','mixed'],default='color');args=parser.parse_args()
    preds=read(ROOT/'Dance_Corrected_Pilot/predictions_detect.json');labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];ff=features(preds,seeds);baseline=read(ROOT/'Dance_Tracking_Pilot/tracks.json')['roster'];val=[165,175,185,195,285,299];results=[];out=OUT/'tracking';settings=read(ROOT/'Dance_Tracking_Pilot/settings.json');start=time.perf_counter()
    if args.feature in ['deep','mixed']:
        deep=read(out/'deep_features.json')
        if args.feature=='deep':ff=deep
        else:
            combine=lambda a,b:(np.r_[a,b]/np.sqrt(2)).tolist()
            ff={'seeds':[combine(a,b) for a,b in zip(ff['seeds'],deep['seeds'])],'frames':{k:[combine(a,b) for a,b in zip(row,deep['frames'][k])] for k,row in ff['frames'].items()}}
            write(out/'mixed_features.json',ff)
    for gw,aw,gallery in itertools.product([0,.15,.35],[0,.15,.3,.5],[False,True]):
        if aw==0 and gallery:continue
        cfg={**settings,'ground_weight':gw,'appearance_weight':aw,'initial_gallery':gallery};name=f'{args.feature}_g{gw}_a{aw}_initial{int(gallery)}';data=run(preds,seeds,ff,cfg);score=core.evaluate(data,labels,val);results.append({'name':name,'config':cfg,'validation':score});write(out/f'{name}_tracks.json',data)
        print(name,'correct IDs',score['correct_initial_ids'],'matched',score['matched_boxes'],'wrong',score['wrong_initial_ids'],flush=True);write(out/f'{args.feature}_results.json',results)
    chosen=max(results,key=lambda r:(r['validation']['correct_initial_ids'],r['validation']['matched_boxes']));write(out/f'{args.feature}_selection.json',{'scope':'Development selection on six later reviewed frames. Same detector cache and first-frame20 seeds. No later reference coordinates used for association.','baseline':core.evaluate(baseline,labels,val),'chosen':chosen,'seconds':time.perf_counter()-start});print('CHOSEN',chosen['name'],flush=True)

if __name__=='__main__':main()
