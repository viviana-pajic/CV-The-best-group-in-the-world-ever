"""Two-detector box fusion, selected using development boxes only."""
import itertools,argparse,cv2,numpy as np
from detector_probe import ROOT,OUT,VAL,read,write,merge,score,match_boxes
from tracker_lab import color_feature

def fuse(first,second,cfg):
    a=merge(first,.65);b=merge(second,.55);links=match_boxes([o['box'] for o in a],[o['box'] for o in b],cfg['match_iou']);used=set(links.values());out=[]
    for i,o in enumerate(a):
        if i in links:
            q=b[links[i]];alpha=cfg['box_alpha'];box=(1-alpha)*np.array(o['box'])+alpha*np.array(q['box']);out.append({'box':box.tolist(),'score':max(o['score'],q['score']),'sources':2})
        else:out.append(dict(o,sources=1))
    out.extend(dict(o,score=o['score']*cfg['unmatched_scale'],sources=1) for j,o in enumerate(b) if j not in used)
    return merge(out,cfg['merge_iou'])

def select():
    first=read(OUT/'probe/previous_960_raw.json');second=read(OUT/'probe/posture_frozen_960_raw.json');labels=read(ROOT/'Dance_Support_Training_Pilot/labels.json');rows=[]
    for mi,alpha,scale,ni in itertools.product([.4,.6],[.25,.5],[.7,1.],[.65,.75]):
        cfg={'match_iou':mi,'box_alpha':alpha,'unmatched_scale':scale,'merge_iou':ni};pred={k:fuse(first[k],second[k],cfg) for k in first}
        for threshold in [.1,.15,.25,.35,.45,.55,.65]:rows.append({'config':cfg,'confidence':threshold,'validation':score(pred,labels,threshold)})
    rows.sort(key=lambda r:(r['validation']['f1'],r['validation']['matched75']),reverse=True);write(OUT/'probe/fusion_results.json',rows);write(OUT/'probe/fusion_selection.json',rows[0]);print('FUSION',rows[0],flush=True)

def sequence():
    first=read(ROOT/'Dance_Corrected_Pilot/predictions_detect.json');second=read(OUT/'sequences/posture_frozen_960/raw.json');cfg=read(OUT/'probe/fusion_selection.json')['config'];pred={k:fuse(first[k],second[k],cfg) for k in first};out=OUT/'sequences/fusion_960';out.mkdir(exist_ok=True);write(out/'raw.json',pred);cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));seeds=read(ROOT/'Dance_Corrected_Pilot/labels.json')['0'];ff={'seeds':[],'frames':{}}
    for k in range(300):
        ok,im=cap.read();assert ok;ff['frames'][str(k)]=[color_feature(im,o['box']).tolist() for o in pred[str(k)]]
        if not k:ff['seeds']=[color_feature(im,o['box']).tolist() for o in seeds]
    cap.release();write(out/'color_features.json',ff);chosen=read(OUT/'probe/fusion_selection.json');write(OUT/'probe/fusion_results_tracking.json',[{'imgsz':960,'merge_iou':.99,**chosen}]);print('Fused300 development frames',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',action='store_true');a=p.parse_args();select()
    if a.sequence:sequence()
