"""Generic ImageNet appearance embeddings; no choreography identity labels used."""
import sys,time,argparse
from pathlib import Path
import torch,cv2,numpy as np
from torchvision.models import resnet18,ResNet18_Weights
from torch import nn
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
from evaluate_support import read,write

def crop_tensor(im,box):
    x1,y1,x2,y2=map(round,box);x1=max(0,x1);x2=min(1280,max(x1+1,x2));y1=max(0,y1);y2=min(720,max(y1+1,y2));p=cv2.resize(im[y1:y2,x1:x2],(96,192));p=p[:,:,::-1].transpose(2,0,1).astype(np.float32)/255;return torch.from_numpy((p-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None])

@torch.inference_mode()
def main(device):
    torch.set_num_threads(4);pred=read(ROOT/'Dance_Corrected_Pilot/predictions_detect.json');seeds=read(ROOT/'Dance_Corrected_Pilot/labels.json')['0'];m=resnet18(weights=ResNet18_Weights.DEFAULT);net=nn.Sequential(*list(m.children())[:-1]).to(device).eval();cache={'seeds':[],'frames':{}};cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));start=time.perf_counter()
    for k in range(300):
        ok,im=cap.read();assert ok;rows=pred[str(k)]+(seeds if k==0 else []);x=torch.stack([crop_tensor(im,o['box']) for o in rows]).to(device);feat=net(x).flatten(1);feat=nn.functional.normalize(feat,dim=1).cpu().numpy();cache['frames'][str(k)]=feat[:len(pred[str(k)])].tolist()
        if k==0:cache['seeds']=feat[len(pred[str(k)]):].tolist()
        if k%50==0:print('Appearance embedding',k,'elapsed',round(time.perf_counter()-start,1),flush=True)
    cap.release();write(OUT/'tracking/deep_features.json',cache);write(OUT/'tracking/deep_feature_provenance.json',{'model':'Generic ImageNet ResNet18 pooled512 features, no person ReID or choreography fine-tuning','crop':[96,192],'seconds':time.perf_counter()-start,'device':device,'warning':'This is an appearance heuristic, not a validated person re-identification network.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');args=p.parse_args();main(args.device)
