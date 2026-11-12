"""Development tracking comparison for the two small identity fine-tunes."""
from reid_trial import *
import time

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);out=OUT/'reid';folder=OUT/'detector/sequences/posture_frozen_960';pred=read(folder/'merged.json');labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];baseline=read(folder/'tracking_selection.json');flow=np.load(OUT/'tracking/dense_flow.npz')['flow'].astype(np.float32);results=[]
    for scope in ['fc','tail']:
        directory=out/f'tuned_{scope}'
        while not (directory/'complete.json').exists():time.sleep(5)
        checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=True);cache=directory/'development_features.json'
        if cache.exists():ff=read(cache)
        else:
            net=model();net.load_state_dict(checkpoint['state_dict']);cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));ff={'seeds':[],'frames':{}}
            for k in range(300):
                ok,im=cap.read();assert ok;ff['frames'][str(k)]=extract(net,im,pred[str(k)])
                if not k:ff['seeds']=extract(net,im,seeds)
            cap.release();write(cache,ff);del net;torch.cuda.empty_cache()
        for gallery in ['seed','training_prototype']:
            feature=copy.deepcopy(ff)
            if gallery=='training_prototype':feature['seeds']=[checkpoint['prototypes'][o['id']-1].tolist() for o in seeds]
            for aw in [.3,.5,.7,.9]:
                cfg={**baseline['config'],'appearance_weight':aw};tracks=run_flow(pred,seeds,feature,cfg,flow);score=core.evaluate(tracks,labels,[165,175,185,195,285,299]);name=f'tuned_{scope}_{gallery}_a{aw}';write(out/f'{name}_tracks.json',tracks);r={'name':name,'scope':scope,'gallery':gallery,'config':cfg,'validation':score};results.append(r);print(name,score['correct_initial_ids'],score['wrong_initial_ids'],score['matched_boxes'],flush=True);write(out/'tuned_tracking_results_partial.json',results)
    chosen=max(results,key=lambda r:(r['validation']['correct_initial_ids'],-r['validation']['wrong_initial_ids'],r['validation']['matched_boxes']));write(out/'tuned_tracking_results.json',{'scope':'Development evaluation only. Models train on fourteen already reviewed training frames, with consistent IDs. Prototype gallery is learned only from those training crops.','baseline':baseline,'candidates':results,'chosen':chosen});print('BEST TUNED REID',chosen['name'],chosen['validation'],flush=True)

if __name__=='__main__':main()
