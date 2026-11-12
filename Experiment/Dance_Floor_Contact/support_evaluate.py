"""Frozen-box inference; threshold selected on validation only."""
import json,time,copy,hashlib
from collections import Counter
import cv2,numpy as np,torch
from support_train import SupportNet,roi,crop,tensor,coordinates
from support_label_setup import OUT,TRAIN,VAL,TEST
from corrected_setup import SOURCE,read
from occlusion_pilot import OUT as TRACK_DIR,evaluate,inside
from ankle_pilot import maps,raw_metrics
from compare import stage,bottom,match_boxes
from map_kalman_pilot import frozen_hash,motion_stats,stationary

def write(name,obj):
    def enc(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v))
    (OUT/name).write_text(json.dumps(obj,default=enc,indent=2),encoding='utf-8')

@torch.inference_mode()
def predict(net,im,rows):
    if not rows:return {}
    rr=np.array([roi(o['box']) for o in rows]);batch=torch.stack([tensor(crop(im,r)) for r in rr]).cuda();hm,v=net(batch);p=coordinates(hm).cpu().numpy();p=rr[:,:2]+p*(rr[:,2:]-rr[:,:2]);v=v.sigmoid().cpu().numpy()
    return {o['id']:{'point':p[i].tolist(),'reportability':float(v[i])} for i,o in enumerate(rows)}

def anchor(o,p,cfg,others):
    if p is None:return None,'no_observed_person',[]
    if p['reportability']<cfg['threshold']:return None,'contact_not_reportable',[]
    q=bottom(np.array(o['box'])) if cfg.get('box_ablation') else np.array(p['point'])
    if not inside(stage(q)):return None,'off_mat',[]
    return q,'box_with_learned_visibility' if cfg.get('box_ablation') else 'learned_support',[]

def raw_copy(base,assoc,threshold=0,box=False):
    d=copy.deepcopy(base)
    for k,row in enumerate(d['frames']):
        for o in row:
            p=assoc[k].get(o['id']);q=bottom(np.array(o['box'])) if box else np.array(p['point']) if p else None
            o['anchor_pixel']=q if o['status']=='observed' and q is not None and inside(stage(q)) and (box or p['reportability']>=threshold) else None;o['anchor_reason']='box_bottom' if box else 'learned_support'
    return d

def reportability(data,labels,frames):
    tp=fp=tn=fn=0
    for k in frames:
        gg=[o for o in labels[str(k)] if o['box'] is not None];rr=[o for o in data['frames'][k] if o['status']=='observed'];links=match_boxes([o['box'] for o in gg],[o['box'] for o in rr],.5)
        for i,j in links.items():
            actual=gg[i]['floor'] is not None;positive=rr[j].get('anchor_pixel') is not None
            if actual and positive:tp+=1
            elif actual:fn+=1
            elif positive:fp+=1
            else:tn+=1
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),scope='Spatially matched observed boxes only. Hidden and uncertain grouped as not reportable.')

def oracle_scores(pred,labels,frames,threshold):
    errors={'box_bottom':[],'learned_all':[],'learned_visible':[]};correct=Counter();emitted=Counter();hidden=Counter();refs=0
    for k in frames:
        for o in labels[str(k)]:
            if o['box'] is None:continue
            p=pred[str(k)][str(o['id'])];points={'box_bottom':bottom(np.array(o['box'])),'learned_all':p['point'],'learned_visible':p['point'] if p['reportability']>=threshold else None}
            refs+=o['floor'] is not None
            for name,q in points.items():
                if q is None:continue
                emitted[name]+=1
                if o['floor'] is None:hidden[name]+=1;continue
                e=float(np.linalg.norm(stage(q)-stage(o['floor'])));errors[name].append(e);correct[name]+=e<=3
    return {'scope':'Localization diagnostic with manually reviewed boxes, removing tracker-box mismatch. Same final-section labels, not an independent test.','visible_references':refs,'methods':{m:{'correct_within_3pct':correct[m],'median_error_pct_width':float(np.median(e)) if e else None,'emitted':emitted[m],'emitted_on_hidden_or_uncertain':hidden[m]} for m,e in errors.items()}}

def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    labels=read(OUT/'labels.json');base=read(TRACK_DIR/'tracks.json')['roster'];ckpt=torch.load(OUT/'support_best.pt',map_location='cpu',weights_only=True)
    assert ckpt['label_sha256']==hashlib.sha256((OUT/'labels.json').read_bytes()).hexdigest()
    net=SupportNet(False).cuda().eval();net.load_state_dict(ckpt['state_dict']);assoc=[];reviewed={};cap=cv2.VideoCapture(str(SOURCE/'source.mp4'));start=time.perf_counter()
    for k in range(300):
        ok,im=cap.read();assert ok
        assoc.append(predict(net,im,[o for o in base['frames'][k] if o['status']=='observed']))
        if k in VAL+TEST:reviewed[str(k)]=predict(net,im,[o for o in labels[str(k)] if o['box'] is not None])
        if k%75==0:print('Support inference',k,'/300',flush=True)
    cap.release();seconds=time.perf_counter()-start;write('predictions.json',assoc);write('reviewed_box_predictions.json',reviewed)
    results=[]
    for t in [.1,.25,.4,.5,.6,.75,.9]:
        dd=raw_copy(base,assoc,t);score=raw_metrics(dd,labels,VAL);results.append({'threshold':t,'raw_validation':score,'visibility':reportability(dd,labels,VAL)})
    selected=max(results,key=lambda d:(d['raw_validation']['joint_f1'],d['raw_validation']['correct_within_3pct']));threshold=selected['threshold']
    write('selection.json',{'candidates':results,'threshold':threshold,'rule':'Maximum raw-point joint F1 on four validation frames; tie-break correct point count. F1 penalizes missing references and all emitted proposals, including wrong/hidden points. No test result used.'})
    current=raw_copy(base,assoc,box=True);learned=maps(base,assoc,{'threshold':threshold},anchor);ungated=maps(base,assoc,{'threshold':0},anchor);ablation=maps(base,assoc,{'threshold':threshold,'box_ablation':True},anchor)
    methods={'box_bottom':current,'learned_visible':learned,'learned_all':ungated,'box_learned_visibility':ablation}
    scores={s:{n:{'raw':raw_metrics(d,labels,frames),'map':evaluate(d,labels,frames),'reportability':reportability(d,labels,frames)} for n,d in methods.items()} for s,frames in [('validation',VAL),('development_test',TEST)]}
    # Test labels remain untouched. Record sensitivity to the location tolerance.
    details=[]
    for k in TEST:
        gt=[o for o in labels[str(k)] if o['box'] is not None];rr=[o for o in base['frames'][k] if o['status']=='observed'];links=match_boxes([o['box'] for o in gt],[o['box'] for o in rr],.5)
        for i,g in enumerate(gt):
            if g['floor'] is None:continue
            rec={'frame':k,'reference_id':g['id'],'support_type':g['support_type'],'tracker_id':rr[links[i]]['id'] if i in links else None}
            for name,d in methods.items():
                o=next((o for o in d['frames'][k] if o['id']==rec['tracker_id']),None)
                rec[name+'_raw_error']=float(np.linalg.norm(stage(o['anchor_pixel'])-stage(g['floor']))) if o and o.get('anchor_pixel') is not None else None
                rec[name+'_map_error']=float(np.linalg.norm(np.array(o['map'])-stage(g['floor']))) if o and o.get('map') is not None else None
            details.append(rec)
    write('paired_errors.json',details)
    write('tolerance_sensitivity.json',{str(t):{n:sum(r[n+'_raw_error'] is not None and r[n+'_raw_error']<=t for r in details) for n in methods} for t in [1,2,3,5]})
    write('accuracy.json',scores);write('tracks.json',{'current':current,'learned':learned,'point_only':ungated})
    motion=motion_stats(current,learned);motion['metrics']['learned']=motion['metrics'].pop('kalman')
    stat=stationary(current,learned,labels);stat['learned']=stat.pop('kalman')
    write('motion.json',motion);write('stationary.json',stat)
    pointstat=stationary(current,ungated,labels);pointstat['learned']=pointstat.pop('kalman');write('point_only_stationary.json',pointstat)
    posture_counts={s:dict(Counter(o['support_type'] or o['contact'] for k in kk for o in labels[str(k)])) for s,kk in [('train',TRAIN),('validation',VAL),('test',TEST)]}
    visibility_by_posture=Counter()
    for k in TEST:
        gg=[o for o in labels[str(k)] if o['box'] is not None];rr=[o for o in learned['frames'][k] if o['status']=='observed'];links=match_boxes([o['box'] for o in gg],[o['box'] for o in rr],.5)
        for i,j in links.items():
            if gg[i]['floor'] is not None:
                kind=gg[i]['support_type'];visibility_by_posture[kind+'_matched']+=1;visibility_by_posture[kind+'_accepted']+=rr[j].get('anchor_pixel') is not None
    write('posture_diagnostic.json',{'label_types':posture_counts,'test_matched_visibility':dict(visibility_by_posture),'interpretation':'Visibility rejection depends strongly on posture in this sample. Training/validation do not adequately cover visible kneeling in different rows.'})
    write('coverage.json',{n:dict(Counter(o['map_status'] for row in d['frames'] for o in row)) for n,d in methods.items()})
    reviewed=read(OUT/'reviewed_box_predictions.json');write('reviewed_box_diagnostic.json',oracle_scores(reviewed,labels,TEST,threshold))
    assert all(frozen_hash(base)==frozen_hash(d) for d in methods.values())
    write('provenance.json',{'same_detector_boxes_ids_status':True,'frozen_hash':frozen_hash(base),'labels_sha256':ckpt['label_sha256'],'checkpoint_sha256':hashlib.sha256((OUT/'support_best.pt').read_bytes()).hexdigest(),'inference_seconds':seconds,'fps_including_reviewed_box_diagnostics':300/seconds,'checkpoint_epoch':ckpt['epoch'],'threshold':threshold,'map_filter':'Same median3, bounded EMA0.35, 3-reading jump confirmation and 25-frame hold. Baseline keeps existing overlap-based availability; learned method uses predicted reportability instead.','no_box_bottom_fallback':True,'limits':['Only 12 training images; same dancers and already-reviewed development clip.','Assistant visual labels are approximate; no independent human review.','75 visible support references in five final frames; not the prior ankle benchmark labels.','Fixed detector and IDs retain prior tracking errors.','Reportability is an uncalibrated model score, not proof of contact.','Calibration remains approximate; coordinates in percent of relative mat width.']})
    print(json.dumps({'threshold':threshold,'test':read(OUT/'accuracy.json')['development_test'],'oracle':read(OUT/'reviewed_box_diagnostic.json')},indent=2),flush=True)

if __name__=='__main__':main()
