"""Calibrate learned-feature distance on development, without changing frozen comparisons."""
from reid_trial import *
import time

def main():
    torch.set_num_threads(4);out=OUT/'reid';directory=out/'tuned_tail'
    while not (directory/'development_features.json').exists():time.sleep(5)
    checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=True);ff=read(directory/'development_features.json');folder=OUT/'detector/sequences/posture_frozen_960';pred=read(folder/'merged.json');labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];baseline=read(folder/'tracking_selection.json');flow=np.load(OUT/'tracking/dense_flow.npz')['flow'].astype(np.float32);results=[]
    for gallery,scale,aw in itertools.product(['seed','training_prototype'],[.6,1.0],[.3,.5,.7]):
        feature=copy.deepcopy(ff)
        if gallery=='training_prototype':feature['seeds']=[checkpoint['prototypes'][o['id']-1].tolist() for o in seeds]
        cfg={**baseline['config'],'appearance_weight':aw,'appearance_distance_scale':scale};tracks=run_flow(pred,seeds,feature,cfg,flow);score=core.evaluate(tracks,labels,[165,175,185,195,285,299]);name=f'tuned_tail_{gallery}_scale{scale}_a{aw}';write(out/f'{name}_tracks.json',tracks);results.append({'name':name,'gallery':gallery,'config':cfg,'validation':score});print(name,score['correct_initial_ids'],score['wrong_initial_ids'],score['matched_boxes'],flush=True)
    chosen=max(results,key=lambda r:(r['validation']['correct_initial_ids'],-r['validation']['wrong_initial_ids'],r['validation']['matched_boxes']));write(out/'calibrated_tracking_results.json',{'scope':'Development only; wider cosine-distance scaling for the newly trained representation. Existing frozen pipeline uses default.35 and is unchanged.','candidates':results,'chosen':chosen});print('BEST CALIBRATED',chosen,flush=True)

if __name__=='__main__':main()
