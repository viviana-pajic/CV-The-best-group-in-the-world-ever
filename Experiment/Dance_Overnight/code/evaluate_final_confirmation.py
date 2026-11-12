"""One application of precommitted variants to newly sealed confirmation labels."""
import frozen_holdout as fh
from frozen_holdout import *
import trajectory_fit

def score(tracks,labels,observed_only=False):
    records=[]
    for k,gt in labels.items():
        pp=[o for o in tracks['frames'][int(k)-3250] if o['status']=='observed' or (not observed_only and o['status']=='predicted' and o['age_frames']<=25)];links=match_boxes([o['box'] for o in gt],[o['box'] for o in pp],.5)
        for i,o in enumerate(gt):
            q=pp[links[i]] if i in links else None;p=q['map'] if q else None;state={'proxy':1,'estimated':2,'unknown':0}[q['map_status']] if q else 0;r=record(k,o,p,state,q is not None,q['id'] if q else None)
            if p is not None and o['floor'] is not None:r['error_dx'],r['error_dy']=(np.array(p)-stage(o['floor'])).tolist()
            records.append(r)
    return fh.result(records)

def main():
    torch.set_num_threads(3);cv2.setNumThreads(2);dest=OUT/'final_confirmation';fh.D=OUT/'data/final_confirmation';freeze=read(dest/'frozen_selection.json');sealed=read(fh.D/'sealed_protocol.json');assert hashlib.sha256((fh.D/'labels.json').read_bytes()).hexdigest()==sealed['label_sha256'];assert hashlib.sha256((dest/'frozen_selection.json').read_bytes()).hexdigest()==sealed['selection_sha256'];assert not (dest/'results.json').exists(),'Preserve the first confirmation result.'
    while not (OUT/'late_full_replay/provenance.json').exists():time.sleep(5)
    provenance=read(OUT/'late_full_replay/provenance.json');base=read(OUT/provenance['base_tracks']);pred=read(OUT/provenance['support_predictions']);late=read(OUT/provenance['tracks']);old=read(OUT/'ending_experiment/pipeline/previous_support_full_tracks.json');labels=read(fh.D/'labels.json');boxpred=copy.deepcopy(pred)
    for k,rr in enumerate(base['frames']):
        for o in rr:
            if str(o['id']) in boxpred[k]:boxpred[k][str(o['id'])]['point']=bottom(np.array(o['box'])).tolist()
    z,cf,ids=inputs(base,boxpred,freeze['threshold']);a,st,age=trajectory_fit.fit(z,cf,freeze['settings']);box=export(base,a,st,age,z,ids,freeze['settings']);write(dest/'box_bottom_full_tracks.json',box)
    variants={'Incumbent support':old,'Expanded support':late,'Selected pipeline / box bottom':box};r={'protocol':sealed,'selection':freeze,'scope':'New labeled temporal block in the same choreography. No model, filter or threshold tuning from these labels. Assistant source-only labels, image-local IDs, correlated frames. Includes annotated distinguishable bodies only, not full20-person detection ground truth. All variants use the same full2424-frame temporal extent.','candidates':{},'observed_only':{},'aspect_sensitivity':{}}
    for name,tracks in variants.items():
        r['candidates'][name]=score(tracks,labels);r['observed_only'][name]=score(tracks,labels,True);sensitivity={}
        for ratio in [1.3,1.6,1.9]:
            records=copy.deepcopy(r['candidates'][name]['records'])
            for item in records:
                if item['error_pct'] is not None:item['error_pct']=float(np.hypot(item['error_dx'],item['error_dy']*1.6/ratio))
            sensitivity[str(ratio)]=aggregate(records)
        r['aspect_sensitivity'][name]=sensitivity
    original=r['candidates']['Incumbent support']['all'];expanded=r['candidates']['Expanded support']['all'];passed=expanded['correct_by_tolerance_pct_width']['3']>=original['correct_by_tolerance_pct_width']['3'] and expanded['correct_by_tolerance_pct_width']['1']>original['correct_by_tolerance_pct_width']['1'] and expanded['hidden_measured']<=original['hidden_measured']+1;r['precommitted_improvement_condition_met']=passed
    r['interpretation']=('The expanded pipeline met the precommitted improvement conditions on the new block: it improved the1% count, did not reduce the3% count, and increased false visible reports for hidden contacts by no more than one. This supports using it as the current candidate, with the earlier crowd regression still disclosed.' if passed else 'The expanded pipeline did not meet all precommitted improvement conditions on the new block. Keep the incumbent as the established reference and the expanded model as a seated-pose alternative. This block does not justify calling the late model an overall improvement.')+' Seven correlated frames from the same choreography and assistant reference labels limit the conclusion. Identity switches are not scored here. Reference-label repeatability is comparable to some differences at the strict1% tolerance.'
    r['interpretation']=r['interpretation'].replace('the1%','the 1%').replace('the3%','the 3%').replace('strict1%','strict 1%');write(dest/'results.json',r)
    oldsupport=read(OUT/'round2/frozen_selection.json')['base_pipeline']['support'];oldthreshold=read(OUT/'round2/frozen_selection.json')['selection']['threshold'];oracle={}
    for name,cfg,threshold in [('incumbent',oldsupport,oldthreshold),('expanded',freeze['support'],freeze['threshold'])]:
        pp=fh.oracle_predict(cfg,labels,dest/'oracle'/name);oracle[name]=fh.oracle_score(pp,labels,threshold)
    oracle['box_bottom']=fh.oracle_score(None,labels,0,True);write(dest/'oracle_diagnostic.json',oracle);print('Confirmation complete',r['interpretation'],flush=True)
    for name,scores in r['candidates'].items():print(name,scores['all'],flush=True)

if __name__=='__main__':main()
