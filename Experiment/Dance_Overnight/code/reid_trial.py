"""Compare a pretrained person-ReID representation on existing development identities."""
from tracker_lab import *
from flow_tracker import run as run_flow
import torch,itertools
sys.path.insert(0,str(OUT/'external/osnet'))
from osnet import osnet_x1_0

def model():
    net=osnet_x1_0(num_classes=1,pretrained=False).cuda().eval();state=torch.load(OUT/'external/osnet/reid_weights.pt',map_location='cpu',weights_only=True);state=state.get('state_dict',state);state={k.removeprefix('module.'):v for k,v in state.items()};compatible={k:v for k,v in state.items() if k in net.state_dict() and v.shape==net.state_dict()[k].shape};missing,unexpected=net.load_state_dict(compatible,strict=False);assert set(missing)<=set(['classifier.weight','classifier.bias']) and not unexpected,(missing,unexpected);return net

@torch.inference_mode()
def extract(net,im,rows):
    if not rows:return []
    patches=[]
    for o in rows:
        x1,y1,x2,y2=map(round,o['box']);patch=im[max(0,y1):min(720,y2),max(0,x1):min(1280,x2)];assert patch.size;patch=cv2.cvtColor(cv2.resize(patch,(128,256)),cv2.COLOR_BGR2RGB).astype(np.float32)/255;patch=(patch-np.array([.485,.456,.406]))/np.array([.229,.224,.225]);patches.append(torch.from_numpy(patch.transpose(2,0,1)).float())
    f=net(torch.stack(patches).cuda());f=torch.nn.functional.normalize(f,dim=1);return f.cpu().tolist()

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);out=OUT/'reid';out.mkdir(exist_ok=True);folder=OUT/'detector/sequences/posture_frozen_960';pred=read(folder/'merged.json');color=read(folder/'color_features.json');labels=read(ROOT/'Dance_Corrected_Pilot/labels.json');seeds=labels['0'];cache=out/'development_features.json'
    if cache.exists():deep=read(cache)
    else:
        net=model();cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));deep={'seeds':[],'frames':{}}
        for k in range(300):
            ok,im=cap.read();assert ok;deep['frames'][str(k)]=extract(net,im,pred[str(k)])
            if k==0:deep['seeds']=extract(net,im,seeds)
            if k%100==0:print('ReID features',k,flush=True)
        cap.release();write(cache,deep);del net;torch.cuda.empty_cache()
    baseline=read(folder/'tracking_selection.json');flow=np.load(OUT/'tracking/dense_flow.npz')['flow'].astype(np.float32);results=[]
    for representation in ['reid','mixed']:
        if representation=='reid':ff=deep
        else:
            combine=lambda a,b:(np.r_[a,b]/np.sqrt(2)).tolist();ff={'seeds':[combine(a,b) for a,b in zip(color['seeds'],deep['seeds'])],'frames':{k:[combine(a,b) for a,b in zip(color['frames'][k],deep['frames'][k])] for k in pred}}
        for aw in [.3,.5,.7,.9]:
            cfg={**baseline['config'],'appearance_weight':aw};tracks=run_flow(pred,seeds,ff,cfg,flow);score=core.evaluate(tracks,labels,[165,175,185,195,285,299]);name=f'{representation}_a{aw}';write(out/f'{name}_tracks.json',tracks);results.append({'name':name,'representation':representation,'config':cfg,'validation':score});print(name,score['correct_initial_ids'],score['wrong_initial_ids'],score['matched_boxes'],flush=True)
    chosen=max(results,key=lambda r:(r['validation']['correct_initial_ids'],-r['validation']['wrong_initial_ids'],r['validation']['matched_boxes']));write(out/'development_results.json',{'scope':'Development only. New pretrained ReID architecture, no extra identity labels or training. First reserved and backup results are not untouched tests for later tracking changes.','baseline':baseline,'candidates':results,'chosen':chosen});print('ReID trial complete',chosen['name'],flush=True)

if __name__=='__main__':main()
