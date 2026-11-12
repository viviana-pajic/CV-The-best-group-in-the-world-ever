"""Fixed-threshold development check of repeat seeds and an unweighted ensemble."""
from frozen_holdout import *
from candidate_evaluate import oracle_scores
from evaluate_ending import clean_manifest
import trajectory_fit

def average(rows):
    result={}
    for ident in rows[0]:
        result[ident]={'point':np.mean([r[ident]['point'] for r in rows],axis=0).tolist(),'peak_point':np.mean([r[ident]['peak_point'] for r in rows],axis=0).tolist(),'reportability':float(np.mean([r[ident]['reportability'] for r in rows]))}
    return result

def main():
    while not (OUT/'runs/v9_repeat_wide_s43/complete.json').exists():time.sleep(5)
    torch.set_num_threads(4);cv2.setNumThreads(2);dest=OUT/'seed_repeatability';manifest=clean_manifest();summary={'protocol':read(dest/'protocol.json'),'models':{}}
    chosen_detector=read(OUT/'detector_expanded/tracking_selection.json')['chosen'];base=read(OUT/chosen_detector['tracks']);filtercfg=read(OUT/'round2/frozen_selection.json')['selection']['settings'];data=read(OUT/'data/development_v1.json')['controlled'];cfg0=read(OUT/'final_confirmation/frozen_selection.json')['support'];names=['v6_attention_wide_s17','v9_repeat_wide_s29','v9_repeat_wide_s43'];oracles={};preds={}
    oracles[names[0]]=read(OUT/'ending_experiment/v6_attention_wide_s17_best_joint_crop5flip_predictions.json');preds[names[0]]=read(OUT/'late_full_replay/support/predictions.json')[1170:1470]
    for name in names[1:]:
        while not (OUT/'runs'/name/'complete.json').exists():time.sleep(5)
        net,condition,ck=load_model(OUT/'runs'/name/'best_joint.pt');cfg={**cfg0,'name':name+'_best_joint_crop5flip','run':name,'epoch':ck['epoch']};pp={}
        for clip,d in manifest.items():
            for k in d['val']:
                im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f"frame_{d['source_first']+k:06}.png"));pp[f'{clip}:{k}']=predict_aug(net,im,d['labels'][str(k)],'crop5flip',condition)
        folder=dest/name;folder.mkdir(exist_ok=True);write(folder/'oracle.json',pp);oracles[name]=read(folder/'oracle.json');del net;torch.cuda.empty_cache()
        infer(cfg,base,ROOT/'Dance_Controlled_Test/source.mp4',folder/'controlled','cuda');preds[name]=read(folder/'controlled/predictions.json')
    oracles['ensemble']={k:average([oracles[n][k] for n in names]) for k in oracles[names[0]]};preds['ensemble']=[average([preds[n][k] for n in names]) for k in range(300)]
    for name in names+['ensemble']:
        score=next(r for r in oracle_scores(oracles[name],manifest)['options'] if r['threshold']==.25);z,cf,ids=inputs(base,preds[name],.25);a,st,age=trajectory_fit.fit(z,cf,filtercfg);controlled=identity_score(a,st,data['labels'],data['val'],ids);summary['models'][name]={'oracle':score,'controlled':controlled};write(dest/(name+'_tracks.json'),export(base,a,st,age,z,ids,filtercfg));print(name,score['correct'],score['hidden_emitted'],controlled['correct_by_tolerance_pct_width'],controlled['hidden_measured'],flush=True)
    summary['interpretation']='Fixed development comparison only. These repeats and the three-model average do not replace the earlier frozen, independently confirmed candidate. Ensemble requires three times the support inference.'
    write(dest/'results.json',summary)

if __name__=='__main__':main()
