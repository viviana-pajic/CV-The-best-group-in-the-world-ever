"""Development-only old-first support fallback; preserves frozen confirmation."""
from frozen_holdout import *
from candidate_evaluate import oracle_scores
from evaluate_ending import clean_manifest
import trajectory_fit,itertools

def combine_row(old,new,t_old,t_new,average=False):
    out={}
    for ident,p in old.items():
        q=new.get(ident,p);sp=stage(p['point']);sq=stage(q['point']);vp=p['reportability']>=t_old and 0<=sp[0]<=100 and 0<=sp[1]<=62.5;vq=q['reportability']>=t_new and 0<=sq[0]<=100 and 0<=sq[1]<=62.5
        if vp:
            r=dict(p)
            if average and vq and np.linalg.norm(sp-sq)<=2:r['point']=((np.array(p['point'])+q['point'])/2).tolist();r['reportability']=(p['reportability']+q['reportability'])/2
            r['reportability']=max(.5,r['reportability'])
        elif vq:r={**q,'reportability':max(.5,q['reportability'])}
        else:r={**p,'reportability':0.}
        out[ident]=r
    return out

def main():
    torch.set_num_threads(2);dest=OUT/'fallback_experiment';dest.mkdir(exist_ok=True);manifest=clean_manifest();old=read(OUT/'ending_experiment/v5_attention_s17_best_joint_crop5flip_predictions.json');new=read(OUT/'ending_experiment/v6_attention_wide_s17_best_joint_crop5flip_predictions.json');rows=[]
    for to,tn,avg in itertools.product([.25,.4,.5,.6],[.25,.4,.5,.6],[False,True]):
        pp={k:combine_row(row,new[k],to,tn,avg) for k,row in old.items()};s=next(x for x in oracle_scores(pp,manifest)['options'] if x['threshold']==.5);rows.append({'old_threshold':to,'new_threshold':tn,'average_when_within_2pct':avg,'pooled_oracle':s})
    reference=read(OUT/'ending_experiment/results.json')['chosen']['pooled'];eligible=[r for r in rows if r['pooled_oracle']['joint_f1']>=reference['joint_f1'] and r['pooled_oracle']['hidden_emitted']<=reference['hidden_emitted']+1];eligible.sort(key=lambda r:(r['pooled_oracle']['joint_f1'],-r['pooled_oracle']['hidden_emitted']),reverse=True);write(dest/'oracle_results.json',{'reference':reference,'candidates':rows,'eligible':eligible})
    chosen_detector=read(OUT/'detector_expanded/tracking_selection.json')['chosen'];base=read(OUT/chosen_detector['tracks']);op=read(OUT/'detector_expanded/selected_support/predictions.json');npred=read(OUT/'late_full_replay/support/predictions.json')[1170:1470];frozen=read(OUT/'round2/frozen_selection.json')['selection']['settings'];d=read(OUT/'data/development_v1.json')['controlled'];pipeline=[];baselines={}
    for name,pp,t in [('old_head',op,.5),('expanded_head',npred,.25)]:
        z,cf,ids=inputs(base,pp,t);a,st,age=trajectory_fit.fit(z,cf,frozen);baselines[name]=identity_score(a,st,d['labels'],d['val'],ids)
    for rank,cfg in enumerate(eligible[:8]):
        pp=[combine_row(a,b,cfg['old_threshold'],cfg['new_threshold'],cfg['average_when_within_2pct']) for a,b in zip(op,npred)];z,cf,ids=inputs(base,pp,.5);a,st,age=trajectory_fit.fit(z,cf,frozen);score=identity_score(a,st,d['labels'],d['val'],ids);row={**cfg,'controlled_development':score};pipeline.append(row);write(dest/f'candidate{rank}_tracks.json',export(base,a,st,age,z,ids,frozen));print('FALLBACK',rank,score['correct_by_tolerance_pct_width'],'hidden',score['hidden_measured'],flush=True)
    minimum=max(s['correct_by_tolerance_pct_width']['3'] for s in baselines.values());max_hidden=min(s['hidden_measured'] for s in baselines.values());qualified=[r for r in pipeline if r['controlled_development']['correct_by_tolerance_pct_width']['3']>=minimum and r['controlled_development']['hidden_measured']<=max_hidden];selected=max(qualified,key=lambda r:(r['pooled_oracle']['joint_f1'],r['controlled_development']['correct_by_tolerance_pct_width']['1'])) if qualified else None
    write(dest/'results.json',{'scope':'Development-only fallback combination. Original and expanded heads share the selected expanded detector boxes. No fresh confirmation data enter selection. Fixed2%-width agreement for optional averaging; old-first preference otherwise. Reportability values are floored to0.5 only after a head passes its own threshold; rejected observations are0. These are heuristic filter weights, not calibrated probabilities.','reference_oracle':reference,'baselines':baselines,'candidates':pipeline,'selected':selected});print('Fallback selected:',selected is not None,flush=True)

if __name__=='__main__':main()
