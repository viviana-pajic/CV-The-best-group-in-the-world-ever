"""Restore recent body height for support crops when a detector suddenly loses the legs.

Detection boxes, IDs and evaluation boxes remain unchanged. This is a crop-context
experiment, not an amodal ground-truth claim or a new floor measurement.
"""
from frozen_holdout import *
import trajectory_fit,itertools
from pathlib import Path

def repair(base,ratio=.7,memory=25):
    output=copy.deepcopy(base);reference={};events=[]
    for k,rows in enumerate(output['frames']):
        for o in rows:
            if o['status']!='observed':continue
            b=np.asarray(o['box']);ident=o['id'];old=reference.get(ident)
            if old is not None:
                last,p=old;h=p[3]-p[1];nh=b[3]-b[1];suspect=k-last<=memory and nh<ratio*h and b[3]<p[3]-.15*h and b[1]<p[1]+.25*h
                if suspect:
                    r=b.copy();r[3]=min(720.,b[1]+h);o['detection_box']=o['box'];o['box']=r.tolist();o['support_crop_repaired']=True;events.append({'frame':k,'id':ident,'age_from_reliable_box':k-last,'detection_box':b.tolist(),'support_crop_box':r.tolist()});continue
            reference[ident]=(k,b.copy())
    output['support_crop_repairs']=events;return output

def score(base,pred,threshold,settings):
    z,cf,ids=inputs(base,pred,threshold);a,st,age=trajectory_fit.fit(z,cf,settings);d=read(OUT/'data/development_v1.json')['controlled'];return identity_score(a,st,d['labels'],d['val'],ids),export(base,a,st,age,z,ids,settings)

def main():
    torch.set_num_threads(2);cv2.setNumThreads(2);dest=OUT/'roi_repair';dest.mkdir(exist_ok=True);full=OUT/'ending_experiment/pipeline/support'
    while not (full/'timing.json').exists():time.sleep(5)
    selected=read(OUT/'ending_experiment/results.json')['chosen'];cfg=selected['candidate'];threshold=selected['threshold'];settings=read(OUT/'round2/frozen_selection.json')['selection']['settings'];base=read(OUT/'full_replay/forward_tracks.json')
    for key in ['frames','diagnostics']:base[key]=base[key][:300]
    ref=dest/'development_reference';ref.mkdir(exist_ok=True);pred=read(full/'predictions.json')[1170:1470];write(ref/'base_tracks.json',base);write(ref/'predictions.json',pred);write(ref/'settings.json',cfg);baseline,tracks=score(base,pred,threshold,settings);write(ref/'tracks.json',tracks);results=[]
    for ratio,memory in itertools.product([.55,.7,.8],[25,50]):
        name=f'ratio{ratio}_memory{memory}';folder=dest/name;cropbase=repair(base,ratio,memory);infer(cfg,cropbase,ROOT/'Dance_Controlled_Test/source.mp4',folder,'cuda',reuse=ref);pp=read(folder/'predictions.json');accuracy,tracks=score(base,pp,threshold,settings);write(folder/'tracks.json',tracks);row={'name':name,'ratio':ratio,'memory':memory,'repairs':len(cropbase['support_crop_repairs']),'accuracy':accuracy};results.append(row);write(dest/'development_partial.json',results);print('ROI REPAIR',name,row,flush=True)
    eligible=[r for r in results if r['accuracy']['hidden_measured']<=baseline['hidden_measured'] and r['accuracy']['correct_by_tolerance_pct_width']['3']>=baseline['correct_by_tolerance_pct_width']['3']]
    chosen=max(eligible,key=lambda r:(r['accuracy']['correct_by_tolerance_pct_width']['1'],r['accuracy']['correct_by_tolerance_pct_width']['3'],-r['repairs'])) if eligible else None
    improved=chosen is not None and (chosen['accuracy']['correct_by_tolerance_pct_width']['1']>baseline['correct_by_tolerance_pct_width']['1'] or chosen['accuracy']['correct_by_tolerance_pct_width']['3']>baseline['correct_by_tolerance_pct_width']['3'])
    write(dest/'development_results.json',{'scope':'Development only. Original detector boxes and IDs are preserved for association and scoring. Only the support-model crop may extend downward using recent reliable height. Same expanded head, threshold and trajectory filter for all variants.','baseline':baseline,'candidates':results,'chosen':chosen,'demonstrated_development_gain':improved});print('ROI RESULT',improved,chosen,flush=True)

if __name__=='__main__':main()
