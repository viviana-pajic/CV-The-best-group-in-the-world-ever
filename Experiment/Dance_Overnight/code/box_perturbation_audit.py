"""Fixed development audit: perturb boxes while keeping the source image unchanged."""
from candidate_evaluate import *
from evaluate_ending import clean_manifest
import copy

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    dest=OUT/'box_perturbation';dest.mkdir(exist_ok=True)
    manifest=clean_manifest();specs={'original':[0,0,0,0]}
    for axis,delta in [('x',[1,0,1,0]),('y',[0,1,0,1]),('bottom',[0,0,0,1]),('height',[0,-1,0,1])]:
        for sign in [-1,1]:specs[f'{axis}_{sign*4}']=(np.array(delta)*sign*4).tolist()
    protocol={'scope':'Development-only synthetic crop sensitivity. Same image, +/-4 source-pixel box changes. No physical motion, no tracking, no temporal smoothing. Not a new heldout evaluation or actual temporal jitter measurement. Settings unchanged.','perturbations':specs,'models':{'established':'v5_attention_s17','expanded':'v6_attention_wide_s17'},'mode':'crop5flip','threshold':.25}
    write(dest/'protocol.json',protocol);records=[]
    for name,run in protocol['models'].items():
        net,condition,_=load_model(OUT/'runs'/run/'best_joint.pt')
        for clip,d in manifest.items():
            for k in d['val']:
                im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f"frame_{d['source_first']+k:06}.png"));original=d['labels'][str(k)];pred={}
                for tag,delta in specs.items():
                    rows=copy.deepcopy(original)
                    for o in rows:o['box']=(np.array(o['box'])+delta).tolist()
                    pred[tag]=predict_aug(net,im,rows,'crop5flip',condition)
                for o in original:
                    ident=o['id'];p0=pred['original'][ident];q0=np.array(p0['point']);b0=bottom(np.array(o['box']));v0=p0['reportability']>=.25
                    for tag,delta in specs.items():
                        if tag=='original':continue
                        p=pred[tag][ident];q=np.array(p['point']);b=bottom(np.array(o['box'])+delta);gt=stage(o['floor']) if o['floor'] is not None else None
                        records.append({'model':name,'clip':clip,'frame':k,'id':ident,'perturbation':tag,'visible':gt is not None,'point_shift_pixels':float(np.linalg.norm(q-q0)),'point_shift_stage':float(np.linalg.norm(stage(q)-stage(q0))),'box_shift_pixels':float(np.linalg.norm(b-b0)),'box_shift_stage':float(np.linalg.norm(stage(b)-stage(b0))),'visibility_flip':(p['reportability']>=.25)!=v0,'original_error':float(np.linalg.norm(stage(q0)-gt)) if gt is not None else None,'perturbed_error':float(np.linalg.norm(stage(q)-gt)) if gt is not None else None,'original_box_error':float(np.linalg.norm(stage(b0)-gt)) if gt is not None else None,'perturbed_box_error':float(np.linalg.norm(stage(b)-gt)) if gt is not None else None})
                print(name,clip,k,flush=True)
        del net;torch.cuda.empty_cache()
    results={}
    for name in protocol['models']:
        rr=[r for r in records if r['model']==name];vis=[r for r in rr if r['visible']]
        results[name]={'perturbed_examples':len(rr),'visibility_flips':sum(r['visibility_flip'] for r in rr),'visible_perturbations':len(vis),'original_correct_3':sum(r['original_error']<=3 for r in vis),'perturbed_correct_3':sum(r['perturbed_error']<=3 for r in vis),'original_box_correct_3':sum(r['original_box_error']<=3 for r in vis),'perturbed_box_correct_3':sum(r['perturbed_box_error']<=3 for r in vis)}
        for key in ['point_shift_pixels','point_shift_stage','box_shift_pixels','box_shift_stage']:
            results[name][key]={'median':float(np.median([r[key] for r in rr])),'p90':float(np.percentile([r[key] for r in rr],90))}
        results[name]['visible_only_sensitivity']={key:{'median':float(np.median([r[key] for r in vis])),'p90':float(np.percentile([r[key] for r in vis],90))} for key in ['point_shift_pixels','point_shift_stage','box_shift_pixels','box_shift_stage']}
    write(dest/'records.json',records);write(dest/'results.json',{'protocol':protocol,'models':results});print(json.dumps(results),flush=True)

if __name__=='__main__':main()
