"""Post-test ablations. These explain the first result; they are not new heldout selections."""
from frozen_holdout import *
from support_train import roi

def compact(r):
    a=r['all'];return {k:a[k] for k in ['visible_references','available_visible','correct_by_tolerance_pct_width','median_error_pct','p90_error_pct','hidden_measured']} | {'phase_3pct':{k:v['correct_by_tolerance_pct_width']['3'] for k,v in r['by_phase'].items()}}

def main():
    folder=OUT/'reserved_evaluation/offline';base=read(folder/'base_tracks.json');pred=read(folder/'predictions.json');cfg=read(OUT/'search/frozen_selection.json')['offline']['chosen'];labels=read(D/'labels.json');results={}
    variants={'learned_raw':(pred,{'kind':'raw'}),'learned_filtered':(pred,cfg['settings'])};bp=copy.deepcopy(pred)
    for k,rows in enumerate(base['frames']):
        for o in rows:
            if str(o['id']) in bp[k]:bp[k][str(o['id'])]['point']=bottom(np.array(o['box'])).tolist()
    variants['box_same_visibility_raw']=(bp,{'kind':'raw'});variants['box_same_visibility_filtered']=(bp,cfg['settings'])
    for name,(pp,settings) in variants.items():
        if settings['kind']=='raw':settings={**settings,'hold':0}
        z,cf,ids=inputs(base,pp,cfg['threshold']);a,st,age=filtered(z,cf,settings);tracks=export(base,a,st,age,z,ids,settings);results[name]=track_score(tracks,labels);print(name,compact(results[name]),flush=True)
    coverage=[]
    for k,rows in labels.items():
        pp=[o for o in base['frames'][int(k)-4420] if o['status']=='observed'];links=match_boxes([o['box'] for o in rows],[o['box'] for o in pp],.5)
        for i,o in enumerate(rows):
            if o['floor'] is None:continue
            q=pp[links[i]] if i in links else None;r=roi(q['box']) if q else None;p=np.array(o['floor']);covered=bool(r is not None and (p>=r[:2]).all() and (p<=r[2:]).all())
            coverage.append({'frame':int(k),'id':o['id'],'matched':q is not None,'floor_inside_roi':covered})
    write(OUT/'diagnostics/reserved_ablations.json',{'scope':'Post-test diagnostic, same frozen support threshold and filter. Box anchor uses the same learned visibility, boxes and IDs. No parameter optimization.','results':results,'crop_coverage':coverage});print('CROP',len(coverage),sum(x['matched'] for x in coverage),sum(x['floor_inside_roi'] for x in coverage))

if __name__=='__main__':main()
