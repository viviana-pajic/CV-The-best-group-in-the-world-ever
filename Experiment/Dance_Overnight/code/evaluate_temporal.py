"""Matched temporal experiment, development only; does not alter frozen confirmation."""
from frozen_holdout import *
from evaluate_ending import clean_manifest
from candidate_evaluate import oracle_scores
import trajectory_fit
from pathlib import Path

def main():
    while not (OUT/'runs/v8_temporal5_s17/complete.json').exists():time.sleep(5)
    torch.set_num_threads(4);cv2.setNumThreads(2);dest=OUT/'temporal_experiment';dest.mkdir(exist_ok=True);manifest=clean_manifest();old={k:d for k,d in manifest.items() if k!='ending'};reference=read(OUT/'ending_experiment/results.json')['chosen'];base=read(OUT/'roi_repair/development_reference/base_tracks.json');filtercfg=read(OUT/'round2/frozen_selection.json')['selection']['settings'];data=read(OUT/'data/development_v1.json')['controlled'];summary={'scope':'Development-only matched9-channel control and temporal warm fine-tune. Same data, seed, optimizer and architecture. Radius0 repeats current frame; radius2 sees +/-80ms. No confirmation labels or results enter this experiment. The earlier frozen confirmation remains a separate comparison.','reference':reference,'runs':{}}
    for run,radius in [('v8_repeated_current_s17',0),('v8_temporal2_s17',2),('v8_temporal5_s17',5)]:
        rows=[];folder=dest/run;folder.mkdir(exist_ok=True);images={}
        for clip,d in manifest.items():
            for k in d['val']:
                source=d['source_first']+k;images[(clip,k)]=[cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{f:06}.png')) for f in [source-radius,source,source+radius]]
        seen=set()
        for checkpoint in ['best.pt','best_joint.pt']:
            net,condition,c=load_model(OUT/'runs'/run/checkpoint)
            if c['epoch'] in seen:del net;continue
            seen.add(c['epoch'])
            for mode in ['base','flip','crop5flip']:
                cfg={'name':run+'_'+Path(checkpoint).stem+'_'+mode,'run':run,'checkpoint':checkpoint,'epoch':c['epoch'],'mode':mode,'decoder':'point','temporal_radius':radius};pred={f'{clip}:{k}':predict_aug(net,images[(clip,k)],d['labels'][str(k)],mode,False) for clip,d in manifest.items() for k in d['val']};cache=folder/(cfg['name']+'_oracle.json');write(cache,pred);pred=read(cache);row={'candidate':cfg,'pooled':oracle_scores(pred,manifest),'old':oracle_scores(pred,old)};rows.append(row)
            del net;torch.cuda.empty_cache()
        eligible=[]
        for row in rows:
            for score in row['pooled']['options']:
                t=score['threshold'];oldscore=next(s for s in row['old']['options'] if s['threshold']==t)
                if t>=.25 and oldscore['joint_f1']>=reference['old']['joint_f1']-.02:eligible.append({'candidate':row['candidate'],'threshold':t,'pooled':score,'old':oldscore})
        chosen=max(eligible,key=lambda r:(r['pooled']['joint_f1'],r['old']['joint_f1'],r['threshold'])) if eligible else None;result={'oracle_candidates':rows,'chosen':chosen};summary['runs'][run]=result;write(dest/'results_partial.json',summary)
        if chosen:
            infer(chosen['candidate'],base,ROOT/'Dance_Controlled_Test/source.mp4',folder/'controlled','cuda');pred=read(folder/'controlled/predictions.json');z,cf,ids=inputs(base,pred,chosen['threshold']);a,st,age=trajectory_fit.fit(z,cf,filtercfg);tracks=export(base,a,st,age,z,ids,filtercfg);write(folder/'controlled/tracks.json',tracks);result['controlled_development']=identity_score(a,st,data['labels'],data['val'],ids);print('TEMPORAL DEV',run,chosen['candidate'],chosen['threshold'],{k:result['controlled_development'][k] for k in ['correct_by_tolerance_pct_width','hidden_measured','median_error_pct','p90_error_pct']},flush=True)
        write(dest/'results_partial.json',summary)
    summary['scope']=summary['scope'].replace('matched9-channel','matched 9-channel').replace('Radius0 repeats current frame; radius2 sees +/-80ms.','Radius 0 repeats the current frame; radius 2 sees +/-80 ms and radius 5 sees +/-200 ms.')
    summary['original_controlled_development']=read(OUT/'roi_repair/development_results.json')['baseline'];write(dest/'results.json',summary);print('Temporal development comparison complete',flush=True)

if __name__=='__main__':main()
