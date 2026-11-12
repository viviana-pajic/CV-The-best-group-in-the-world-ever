"""Offline forward/backward consistency using automatic final-frame seeds."""
import copy
from flow_tracker import *
from tracker_lab import color_feature
from filter_lab import inputs,filtered,export

def reverse_flow(images):
    path=OUT/'tracking/dense_flow_reverse.npz'
    if path.exists():return np.load(path)['flow'].astype(np.float32)
    alg=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);prev=None;values=[]
    for im in images[::-1]:
        gray=cv2.resize(cv2.cvtColor(im[280:600,100:1220],cv2.COLOR_BGR2GRAY),(560,160));values.append(np.zeros((160,560,2),np.float32) if prev is None else alg.calc(prev,gray,None));prev=gray
    value=np.array(values);np.savez_compressed(path,flow=value.astype(np.float16));return value

def box_maps(data):
    pp=[{str(o['id']):{'point':core.bottom(np.asarray(o['box'])).tolist(),'reportability':1.} for o in row if o['status']=='observed'} for row in data['frames']];z,cf,ids=inputs(data,pp,0);cfg={'name':'box_proxy_for_tracking_ablation','kind':'ema','median':3,'alpha':.35,'gate':2,'confirm':3,'step_cap':2.5,'hold':25};a,st,age=filtered(z,cf,cfg);return export(data,a,st,age,z,ids,cfg)

def combine(forward,backward,mode):
    result=copy.deepcopy(forward);count=len(forward['frames'])
    for k,row in enumerate(result['frames']):
        back={o['id']:o for o in backward['frames'][count-1-k]};priority={}
        for i,o in enumerate(row):
            q=back.get(o['id']);fo=o['status']=='observed';bo=q is not None and q['status']=='observed';agree=fo and bo and core.overlap(np.array(o['box']),np.array(q['box']))>.5
            if agree:o['identity_evidence']='both_directions';priority[o['id']]=3
            elif fo and bo:
                o['identity_evidence']='direction_conflict';priority[o['id']]=1
                if mode!='union':o.update(status='predicted',age_frames=max(1,o['age_frames']),reason='bidirectional_identity_disagreement')
            elif bo and not fo and mode!='forward_reject':
                replacement=copy.deepcopy(q);replacement.update(identity_evidence='backward_only',reason='offline_backward_detection');row[i]=replacement;priority[o['id']]=1
            else:o['identity_evidence']='forward_only' if fo else 'unobserved';priority[o['id']]=2 if fo else 0
        active=sorted([o for o in row if o['status']=='observed'],key=lambda o:priority[o['id']],reverse=True);accepted=[]
        for o in active:
            if any(core.overlap(np.array(o['box']),np.array(p['box']))>.8 for p in accepted):o.update(status='predicted',age_frames=max(1,o['age_frames']),reason='bidirectional_duplicate_conflict')
            else:accepted.append(o)
        result['diagnostics'][k]['observed']=len(accepted)
    result['bidirectional']={'mode':mode,'end_initialization':'Only final forward observed boxes, automatically assigned IDs; no final-frame ground-truth labels used.','lookahead':'Full clip'}
    return box_maps(result)

def main():
    cv2.setNumThreads(4);pred=read(ROOT/'Dance_Corrected_Pilot/predictions_detect.json');labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];ff=features(pred,seeds);selection=read(OUT/'tracking/flow_selection.json')['chosen'];forward=read(OUT/'tracking'/f"{selection['name']}_tracks.json");cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));images=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        images.append(im)
    cap.release();n=len(images);reverse_pred={str(k):pred[str(n-1-k)] for k in range(n)};reverse_seeds=[o for o in forward['frames'][-1] if o['status']=='observed'];reverse_features={'seeds':[color_feature(images[-1],o['box']).tolist() for o in reverse_seeds],'frames':{str(k):ff['frames'][str(n-1-k)] for k in range(n)}};backward=run(reverse_pred,reverse_seeds,reverse_features,selection['config'],reverse_flow(images));write(OUT/'tracking/backward_tracks.json',backward);results=[]
    for mode in ['union','reject_conflict','forward_reject']:
        data=combine(forward,backward,mode);score=core.evaluate(data,labels,[165,175,185,195,285,299]);write(OUT/'tracking'/f'bidirectional_{mode}_tracks.json',data);results.append({'mode':mode,'validation':score});print(mode,score['correct_initial_ids'],score['matched_boxes'],score['wrong_initial_ids'],flush=True)
    write(OUT/'tracking/bidirectional_results.json',{'scope':'Development ablation. Backward seeds come from automatic forward results; no late manual corrections.','variants':results})

if __name__=='__main__':main()
