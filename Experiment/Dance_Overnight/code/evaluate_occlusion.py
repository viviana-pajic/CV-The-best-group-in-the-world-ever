"""Matched synthetic-occlusion benchmark, using known pre-mask targets."""
from train_occlusion import OccludedData
from train_support import Data,read,write,OUT,torch,np,stage,roi
from candidate_evaluate import load_model,coordinates,bottom
from torch.utils.data import DataLoader
from pathlib import Path
import time

@torch.inference_mode()
def evaluate(net,dataset,masked):
    rows=[];offset=0
    for batch in DataLoader(dataset,batch_size=32):
        x,p,v,r,posture=batch[:5];point_valid=batch[5] if masked else v;hm,vl=net(x);q=coordinates(hm).numpy();rr=r.numpy();targets=p.numpy();scores=vl.sigmoid().numpy();points=rr[:,:2]+q*(rr[:,2:]-rr[:,:2]);reference=rr[:,:2]+targets*(rr[:,2:]-rr[:,:2])
        for i in range(len(x)):
            key,o=dataset.rows[offset+i];estimate=stage(points[i]);valid=bool(point_valid[i]);b=np.asarray(o['box']);partial=b.copy();partial[3]=b[1]+.65*(b[3]-b[1]);rpartial=roi(partial);target=np.asarray(o['floor']) if valid else None;rows.append({'clip':key[0],'frame':key[1],'id':o['id'],'target_known':valid,'visible':bool(v[i]),'score':float(scores[i]),'in_mat':bool(0<=estimate[0]<=100 and 0<=estimate[1]<=62.5),'error':float(np.linalg.norm(estimate-stage(reference[i]))) if valid else None,'box_bottom_error':float(np.linalg.norm(stage(bottom(b))-stage(target))) if valid else None,'target_inside_synthetic_partial_roi':bool(np.all(target>=rpartial[:2]) and np.all(target<=rpartial[2:])) if valid else None})
        offset+=len(x)
    result={}
    for name,selected in [('all',rows),('original_development',[r for r in rows if r['clip']!='ending']),('ending_development',[r for r in rows if r['clip']=='ending'])]:
        known=[r for r in selected if r['target_known']];options=[]
        for threshold in [.1,.25,.4,.5,.6,.75,.9]:
            emitted=[r for r in selected if r['score']>=threshold and r['in_mat']];good=[r for r in emitted if r['visible'] and r['error']<=3];refs=sum(r['visible'] for r in selected);options.append({'threshold':threshold,'correct3':len(good),'visible_references':refs,'emitted':len(emitted),'false_visible':sum(not r['visible'] for r in emitted),'joint_f1':2*len(good)/max(refs+len(emitted),1)})
        result[name]={'known_point_references':len(known),'unconditional_correct1':sum(r['in_mat'] and r['error']<=1 for r in known),'unconditional_correct3':sum(r['in_mat'] and r['error']<=3 for r in known),'median_error_pct':float(np.median([r['error'] for r in known])),'p90_error_pct':float(np.percentile([r['error'] for r in known],90)),'visibility_options':options,'best_clean_joint':max(options,key=lambda r:r['joint_f1']) if not masked else None,'known_masked_reported_visible_at_05':sum(r['score']>=.5 and r['in_mat'] for r in known) if masked else None,'full_box_bottom_correct3':sum(r['box_bottom_error']<=3 for r in known),'full_box_bottom_median_error':float(np.median([r['box_bottom_error'] for r in known])),'known_targets_inside_roi_if_box_truncated_to_65pct_height':sum(r['target_inside_synthetic_partial_roi'] for r in known)}
    return result

def main():
    torch.set_num_threads(2);dest=OUT/'occlusion_experiment';dest.mkdir(exist_ok=True)
    while not (OUT/'runs/v7_occlusion_p0.6_s17/complete.json').exists():time.sleep(5)
    manifest=read(OUT/'data/development_ending.json');clean=Data(manifest,'val',ignore_uncertain=True);masked=OccludedData(manifest,'val',ignore_uncertain=True,probability=1.,deterministic=True);models=[('v5_attention_s17','best_joint.pt'),('v6_attention_s17','best_joint.pt')]
    for run in ['v7_occlusion_p0.3_s17','v7_occlusion_p0.6_s17']:
        for ck in ['best_joint.pt','best_masked.pt']:
            if (OUT/'runs'/run/ck).exists():models.append((run,ck))
    results=[]
    for run,ck in models:
        net,_,checkpoint=load_model(OUT/'runs'/run/ck,'cpu');row={'run':run,'checkpoint':ck,'epoch':checkpoint['epoch'],'clean':evaluate(net,clean,False),'synthetic_occlusion':evaluate(net,masked,True)};results.append(row);write(dest/'results_partial.json',results);print('OCCLUSION',run,ck,'cleanF1',round(row['clean']['all']['best_clean_joint']['joint_f1'],3),'masked3%',row['synthetic_occlusion']['all']['unconditional_correct3'],'/',row['synthetic_occlusion']['all']['known_point_references'],flush=True);del net
    write(dest/'results.json',{'scope':'All candidates use one unaugmented crop per dancer and identical deterministic masks on the same development images. Synthetic masks retain known original coordinates; real hidden coordinates remain unknown. This is not a real-occlusion or heldout accuracy claim.','important_limitation':'Original full-body boxes remain available after masking and retain geometric information about the floor. The box-bottom control quantifies that shortcut. Real partial detections can crop the true contact out of the representable heatmap region; the hypothetical65%-height coverage count illustrates this, not a measured real occlusion rate.','reference':'https://arxiv.org/abs/1708.04552','candidates':results})

if __name__=='__main__':main()
