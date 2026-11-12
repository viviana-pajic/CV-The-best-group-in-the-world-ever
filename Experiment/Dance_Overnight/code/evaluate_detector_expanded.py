"""Detector and tracking development comparison for the added full-frame labels."""
from detector_probe import *
from tracker_lab import color_feature,core
from flow_tracker import run as run_flow
from pipeline_infer import infer
from filter_lab import inputs,export
from pipeline_eval import identity_score
import trajectory_fit,itertools
ROOT_OUT=ROOT/'Dance_Overnight';DEST=ROOT_OUT/'detector_expanded'

def detection_score(raw,labels,frames,merge_iou,confidence):
    refs=predictions=matched50=matched75=0;rows=[]
    for k in frames:
        gt=[o['box'] for o in labels[str(k)] if o['box'] is not None and not o.get('outside')];pp=[o['box'] for o in merge(raw[str(k)],merge_iou) if o['score']>=confidence];n=len(match_boxes(gt,pp,.5));m=len(match_boxes(gt,pp,.75));refs+=len(gt);predictions+=len(pp);matched50+=n;matched75+=m;rows.append({'source_frame':k,'refs':len(gt),'predictions':len(pp),'matched50':n,'matched75':m})
    return {'references':refs,'predictions':predictions,'matched50':matched50,'matched75':matched75,'f1':2*matched50/max(refs+predictions,1),'per_frame':rows}

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);runs=['diverse_frozen','diverse_warm','diverse_brightness']
    while not (DEST/'runs/diverse_brightness/complete.json').exists():time.sleep(5)
    labels0=read(ROOT/'Dance_Support_Training_Pilot/labels.json');labels={str(4420+k):labels0[str(k)] for k in VAL};labels.update(read(ROOT_OUT/'data/detector_expansion/reviewed_labels.json'));frames=[4420+k for k in VAL]+[3640];images={k:cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{k:06}.png')) for k in frames};weights={'previous_frozen':ROOT_OUT/'detector/runs/posture_frozen/weights/best.pt',**{name:DEST/'runs'/name/'weights/best.pt' for name in runs}};probes={};(DEST/'probe').mkdir(exist_ok=True)
    for name,path in weights.items():
        model=YOLO(path);raw={str(k):candidates(model,images[k],960,'0') for k in frames};write(DEST/'probe'/f'{name}_raw.json',raw);scores=[]
        for mi,conf in itertools.product([.55,.65,.75],[.1,.15,.25,.35,.45]):scores.append({'merge_iou':mi,'confidence':conf,'pooled':detection_score(raw,labels,frames,mi,conf),'original_development':detection_score(raw,labels,frames[:-1],mi,conf),'extra_floor_frame':detection_score(raw,labels,[3640],mi,conf)})
        scores.sort(key=lambda x:(x['pooled']['f1'],x['pooled']['matched75']),reverse=True);probes[name]={'best':scores[0],'candidates':scores};write(DEST/'probe/results.json',probes);print('PROBE',name,scores[0],flush=True);del model;torch.cuda.empty_cache()
    flow=np.load(ROOT_OUT/'tracking/dense_flow.npz')['flow'].astype(np.float32);labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];basecfg=read(ROOT_OUT/'detector/sequences/posture_frozen_960/tracking_selection.json')['config'];tracking=[]
    for name in runs:
        folder=DEST/'sequences'/name;folder.mkdir(parents=True,exist_ok=True);rawpath=folder/'raw.json';featpath=folder/'raw_color.json'
        if rawpath.exists() and featpath.exists():raw=read(rawpath);feat=read(featpath)
        else:
            model=YOLO(weights[name]);cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));raw={};feat={'seeds':[],'frames':{}}
            for k in range(300):
                ok,im=cap.read();assert ok;raw[str(k)]=candidates(model,im,960,'0');feat['frames'][str(k)]=[color_feature(im,o['box']).tolist() for o in raw[str(k)]]
                if k==0:feat['seeds']=[color_feature(im,o['box']).tolist() for o in seeds]
            cap.release();write(rawpath,raw);write(featpath,feat);del model;torch.cuda.empty_cache()
        nms=probes[name]['best']['merge_iou'];pred={k:merge(row,nms) for k,row in raw.items()};ff={'seeds':feat['seeds'],'frames':{}}
        for k,row in pred.items():
            lookup={tuple(o['box']):f for o,f in zip(raw[k],feat['frames'][k])};ff['frames'][k]=[lookup[tuple(o['box'])] for o in row]
        write(folder/'merged.json',pred);rows=[]
        for high,aw in itertools.product([.1,.15,.25],[.3,.5,.7]):
            cfg={**basecfg,'high_conf':high,'low_conf':.06,'appearance_weight':aw};tracks=run_flow(pred,seeds,ff,cfg,flow);score=core.evaluate(tracks,labels,VAL);key=f'h{high}_a{aw}';write(folder/f'{key}_tracks.json',tracks);row={'name':name+'_'+key,'model':name,'config':cfg,'detector_merge_iou':nms,'validation':score,'tracks':str((folder/f'{key}_tracks.json').relative_to(ROOT_OUT))};rows.append(row);tracking.append(row);print('TRACK',row['name'],score['correct_initial_ids'],score['wrong_initial_ids'],flush=True)
        write(folder/'tracking_results.json',rows)
    chosen=max(tracking,key=lambda r:(r['validation']['correct_initial_ids'],-r['validation']['wrong_initial_ids'],r['validation']['matched_boxes']));write(DEST/'tracking_selection.json',{'scope':'Development-only detector/association comparison. Later already inspected sections can only be diagnostic.','chosen':chosen,'candidates':tracking,'baseline':read(ROOT_OUT/'detector/sequences/posture_frozen_960/tracking_selection.json')});print('EXPANDED DETECTOR CHOSEN',chosen,flush=True)
    frozen=read(ROOT_OUT/'round2/frozen_selection.json');base=read(ROOT_OUT/chosen['tracks']);folder=DEST/'selected_support';infer(frozen['base_pipeline']['support'],base,ROOT/'Dance_Controlled_Test/source.mp4',folder,'cuda');pred=read(folder/'predictions.json');cfg=frozen['selection'];z,cf,ids=inputs(base,pred,cfg['threshold']);a,st,age=trajectory_fit.fit(z,cf,cfg['settings']);tracks=export(base,a,st,age,z,ids,cfg['settings']);write(folder/'tracks.json',tracks);d=read(ROOT_OUT/'data/development_v1.json')['controlled'];write(folder/'development_score.json',identity_score(a,st,d['labels'],d['val'],ids));print('Expanded detector support development',identity_score(a,st,d['labels'],d['val'],ids),flush=True)

if __name__=='__main__':main()
