"""Tune association operating points for a new detector on development only."""
import argparse,itertools
from flow_tracker import *
from detector_probe import merge

def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--imgsz',type=int,default=960);a=p.parse_args();folder=OUT/'detector/sequences'/f'{a.model}_{a.imgsz}';raw=read(folder/'raw.json');features_raw=read(folder/'color_features.json');probe=read(OUT/'detector/probe'/f'{a.model}_results_tracking.json' if a.model=='fusion' else OUT/'detector/probe'/f'{a.model}_results.json');nms=next(r for r in probe if r['imgsz']==a.imgsz)['merge_iou'];pred={k:merge(row,nms) for k,row in raw.items()}
    feat={'seeds':features_raw['seeds'],'frames':{}}
    for k,row in pred.items():
        index={tuple(o['box']):f for o,f in zip(raw[k],features_raw['frames'][k])};feat['frames'][k]=[index[tuple(o['box'])] for o in row]
    write(folder/'merged.json',pred);write(folder/'merged_color.json',feat);flow=prepare_flow();labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];cfgbase=read(OUT/'tracking/flow_selection.json')['chosen']['config'];results=[]
    for high,low,aw in itertools.product([.15,.25],[.06,.10],[.3,.5,.7]):
        cfg={**cfgbase,'high_conf':high,'low_conf':low,'appearance_weight':aw};name=f'{a.model}_h{high}_l{low}_a{aw}';data=run(pred,seeds,feat,cfg,flow);score=core.evaluate(data,labels,[165,175,185,195,285,299]);results.append({'name':name,'config':cfg,'validation':score,'detector_merge_iou':nms});write(folder/f'{name}_tracks.json',data);write(folder/'tracking_results.json',results);print(name,'correct',score['correct_initial_ids'],'matched',score['matched_boxes'],'wrong',score['wrong_initial_ids'],flush=True)
    chosen=max(results,key=lambda r:(r['validation']['correct_initial_ids'],r['validation']['matched_boxes']));write(folder/'tracking_selection.json',chosen);print('CHOSEN',chosen['name'],flush=True)

if __name__=='__main__':main()
