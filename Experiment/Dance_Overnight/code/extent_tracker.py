"""Test temporal box-extent consistency on existing detector candidates."""
from flow_tracker import *

def main():
    cv2.setNumThreads(4);flow=prepare_flow();preds=read(ROOT/'Dance_Corrected_Pilot/predictions_detect.json');labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];ff=features(preds,seeds);base=read(OUT/'tracking/flow_selection.json')['chosen'];results=[]
    for ratio,rise,growth in itertools.product([.65,.75,.85],[.12,.2],[None,1.6]):
        cfg={**base['config'],'lower_body_ratio':ratio,'bottom_rise_fraction':rise,'growth_ratio':growth,'extent_gate_frames':3};name=f'extent_r{ratio}_b{rise}_g{growth}';data=run(preds,seeds,ff,cfg,flow);score=core.evaluate(data,labels,[165,175,185,195,285,299]);dense=[]
        for k in range(165,196):
            o=next(o for o in data['frames'][k] if o['id']==18);dense.append({'frame':k,'status':o['status'],'box':o['box']})
        results.append({'name':name,'config':cfg,'validation':score,'id18_motion_window':dense});write(OUT/'tracking'/f'{name}_tracks.json',data);write(OUT/'tracking/extent_results.json',results)
        print(name,score['correct_initial_ids'],score['matched_boxes'],score['wrong_initial_ids'],'ID18@169',dense[4]['status'],flush=True)
    chosen=max(results,key=lambda r:(r['validation']['correct_initial_ids'],r['validation']['matched_boxes']));write(OUT/'tracking/extent_selection.json',{'scope':'Development extension after a two-frame truncation error found between sparse validation snapshots. Dense-window boxes retained for manual review; no frame-specific correction hardcoded.','chosen':chosen,'motion_baseline':base})

if __name__=='__main__':main()
