"""Compare old and expanded support models on development labels only."""
from candidate_evaluate import *
import copy

def clean_manifest():
    data=read(OUT/'data/development_ending.json')
    for d in data.values():
        d['labels']={k:[o for o in row if o['contact']!='uncertain'] for k,row in d['labels'].items()}
    return data

def main(runs=None,output_folder='ending_experiment'):
    torch.set_num_threads(3);cv2.setNumThreads(2);dest=OUT/output_folder;dest.mkdir(exist_ok=True);manifest=clean_manifest();old={k:d for k,d in manifest.items() if k!='ending'};ending={'ending':manifest['ending']};images={(clip,k):cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f"frame_{d['source_first']+k:06}.png")) for clip,d in manifest.items() for k in d['val']};rows=[]
    runs=runs or ['v5_attention_s17','v5_mask_s17','v5_mask_attention_s17','v6_attention_s17','v6_attention_s29','v6_mask_attention_s17','v6_attention_wide_s17']
    for run in runs:
        folder=OUT/'runs'/run;assert (folder/'complete.json').exists(),run;seen=set()
        for ck in ['best.pt','best_joint.pt']:
            net,condition,checkpoint=load_model(folder/ck)
            if checkpoint['epoch'] in seen:del net;continue
            seen.add(checkpoint['epoch'])
            for mode in ['base','flip','crop5flip']:
                name=f'{run}_{Path(ck).stem}_{mode}';cache=dest/f'{name}_predictions.json';start=time.perf_counter()
                if cache.exists():pred=read(cache)
                else:
                    pred={f'{clip}:{k}':predict_aug(net,images[(clip,k)],d['labels'][str(k)],mode,condition) for clip,d in manifest.items() for k in d['val']};write(cache,pred);pred=read(cache)
                cfg={'name':name+'_point','run':run,'checkpoint':ck,'epoch':checkpoint['epoch'],'mode':mode,'decoder':'point'};r={**cfg,'old_development':oracle_scores(pred,old),'ending_development':oracle_scores(pred,ending),'pooled_development':oracle_scores(pred,manifest),'seconds':time.perf_counter()-start};rows.append(r);write(dest/'results_partial.json',rows);print(name,'old',round(r['old_development']['selected']['joint_f1'],3),'ending',round(r['ending_development']['selected']['joint_f1'],3),flush=True)
            del net;torch.cuda.empty_cache()
    # A changed head must preserve the older postures at the SAME threshold used for ending.
    reference=next(r for r in rows if r['name']=='v5_attention_s17_best_joint_crop5flip_point');candidates=[]
    for r in rows:
        for item in r['pooled_development']['options']:
            t=item['threshold'];oldscore=next(x for x in r['old_development']['options'] if x['threshold']==t);endscore=next(x for x in r['ending_development']['options'] if x['threshold']==t)
            if t<.25 or oldscore['joint_f1']<reference['old_development']['selected']['joint_f1']-.02:continue
            candidates.append({'candidate':{k:r[k] for k in ['name','run','checkpoint','epoch','mode','decoder']},'threshold':t,'pooled':item,'old':oldscore,'ending':endscore})
    chosen=max(candidates,key=lambda r:(r['pooled']['joint_f1'],r['old']['joint_f1'],r['threshold'])) if candidates else None
    write(dest/'results.json',{'scope':'Development-only supplemental data experiment. Four extra ending training frames and two correlated ending validation frames; no new heldout claim. Earlier frozen results remain separate.','reference':reference,'candidates':rows,'eligible':candidates,'chosen':chosen});print('ENDING CHOSEN',chosen,flush=True)

if __name__=='__main__':main()
