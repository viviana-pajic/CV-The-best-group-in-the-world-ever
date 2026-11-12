"""Compare support checkpoints, crop averaging and decoders on development only."""
import sys,json,time,argparse,hashlib
from pathlib import Path
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';sys.path.insert(0,str(ROOT/'Dance_Floor_Contact'))
from support_train import SupportNet,roi,crop,tensor,coordinates
from conditioned_support import ConditionedSupportNet,AttentionSupportNet,target_mask
from compare import stage,bottom
from evaluate_support import read,write

def load_model(path,device='cuda'):
    c=torch.load(path,map_location='cpu',weights_only=True);settings=c.get('settings',{});condition=settings.get('target_box_mask',False)
    if settings.get('temporal_input'):
        from conditioned_support import TemporalSupportNet
        net=TemporalSupportNet().to(device).eval()
    else:net=(AttentionSupportNet(False,condition) if settings.get('contact_attention') else ConditionedSupportNet(False) if condition else SupportNet(False)).to(device).eval()
    net.load_state_dict(c['state_dict']);return net,condition,c

@torch.inference_mode()
def predict_aug(net,im,rows,mode='base',condition=False):
    if not rows:return {}
    examples=[];meta=[]
    for o in rows:
        r0=roi(o['box']);size=r0[2:]-r0[:2];center=(r0[:2]+r0[2:])/2
        specs=[(1,0,0)] if 'crop5' not in mode else [(1,0,0),(.92,0,0),(1.08,0,0),(1,0,-.035),(1,0,.035)]
        for scale,dx,dy in specs:
            mid=center+np.array([dx,dy])*size;r=np.r_[mid-size*scale/2,mid+size*scale/2].astype(np.float32);r[:2]=np.maximum(r[:2],0);r[2:]=np.minimum(r[2:],[1280,720]);patch=[crop(frame,r) for frame in im] if isinstance(im,list) else crop(im,r)
            for flip in ([False,True] if 'flip' in mode else [False]):
                x=torch.cat([tensor(p[:,::-1].copy() if flip else p) for p in patch],0) if isinstance(patch,list) else tensor(patch[:,::-1].copy() if flip else patch)
                if condition:x=torch.cat([x,target_mask(o['box'],r,flip)],0)
                examples.append(x);meta.append((o['id'],r,flip))
    result={o['id']:[] for o in rows}
    for start in range(0,len(examples),64):
        hm,vl=net(torch.stack(examples[start:start+64]).to(next(net.parameters()).device));soft=coordinates(hm);n,h,w=hm.shape;flat=hm.flatten(1);imax=flat.argmax(1);yy,xx=torch.meshgrid(torch.arange(h,device=hm.device),torch.arange(w,device=hm.device),indexing='ij');mask=((xx[None]-(imax%w)[:,None,None]).abs()<=2)&((yy[None]-(imax//w)[:,None,None]).abs()<=2);peak=coordinates(hm.masked_fill(~mask,-1e5));soft=soft.cpu().numpy();peak=peak.cpu().numpy();v=vl.sigmoid().cpu().numpy()
        for i,(ident,r,flip) in enumerate(meta[start:start+64]):
            if flip:soft[i,0]=1-soft[i,0];peak[i,0]=1-peak[i,0]
            result[ident].append((r[:2]+soft[i]*(r[2:]-r[:2]),r[:2]+peak[i]*(r[2:]-r[:2]),float(v[i])))
    return {i:{'point':np.mean([x[0] for x in values],axis=0).tolist(),'peak_point':np.mean([x[1] for x in values],axis=0).tolist(),'reportability':float(np.mean([x[2] for x in values])),'augmentation_rms_pixels':float(np.sqrt(np.mean(np.var(np.array([x[0] for x in values]),axis=0))))} for i,values in result.items()}

def oracle_scores(pred,manifest,decoder='point'):
    rr=[]
    for clip,d in manifest.items():
        for k in d['val']:
            for o in d['labels'][str(k)]:
                if o['box'] is None:continue
                p=pred[f'{clip}:{k}'][str(o['id'])];q=p[decoder];gt=stage(o['floor']) if o['floor'] is not None else None
                rr.append({'visible':gt is not None,'error':float(np.linalg.norm(stage(q)-gt)) if gt is not None else None,'box_error':float(np.linalg.norm(stage(bottom(np.array(o['box'])))-gt)) if gt is not None else None,'score':p['reportability'],'kind':o['support_type'] or o['contact'],'in_mat':bool(0<=stage(q)[0]<=100 and 0<=stage(q)[1]<=62.5)})
    options=[]
    for t in [0,.1,.25,.4,.5,.6,.75,.9]:
        visible=sum(r['visible'] for r in rr);em=[r for r in rr if r['score']>=t and r['in_mat']];good=[r for r in em if r['visible'] and r['error']<=3];bytype={}
        for kind in sorted({r['kind'] for r in rr}):
            items=[r for r in rr if r['kind']==kind];bytype[kind]={'references':len(items),'emitted':sum(r['score']>=t and r['in_mat'] for r in items),'correct':sum(r['visible'] and r['error']<=3 and r['score']>=t and r['in_mat'] for r in items)}
        options.append({'threshold':t,'refs':visible,'emitted':len(em),'correct':len(good),'joint_f1':2*len(good)/(visible+len(em)),'hidden_emitted':sum(not r['visible'] for r in em),'median_error':float(np.median([r['error'] for r in em if r['visible']])) if any(r['visible'] for r in em) else None,'by_type':bytype})
    return {'selected':max(options,key=lambda r:(r['joint_f1'],r['correct'])),'options':options,'box_baseline':{'refs':sum(r['visible'] for r in rr),'correct':sum(r['visible'] and r['box_error']<=3 for r in rr),'median_error':float(np.median([r['box_error'] for r in rr if r['visible']]))}}

def validation(pattern):
    manifest=read(OUT/'data/development_v2.json');images={(clip,k):cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f"frame_{d['source_first']+k:06}.png")) for clip,d in manifest.items() for k in d['val']};summary=[];out=OUT/'candidates';out.mkdir(exist_ok=True)
    for folder in sorted((OUT/'runs').glob(pattern)):
        if not (folder/'complete.json').exists():continue
        seen=set()
        for ck in ['best.pt','best_joint.pt']:
            path=folder/ck
            if not path.exists():continue
            net,condition,checkpoint=load_model(path)
            if checkpoint['epoch'] in seen:del net;continue
            seen.add(checkpoint['epoch'])
            for mode in ['base','flip','crop5','crop5flip']:
                name=f'{folder.name}_{Path(ck).stem}_{mode}';cache=out/f'{name}_validation.json';start=time.perf_counter()
                if cache.exists():pred=read(cache)
                else:
                    pred={}
                    for clip,d in manifest.items():
                        for k in d['val']:pred[f'{clip}:{k}']=predict_aug(net,images[(clip,k)],[o for o in d['labels'][str(k)] if o['box'] is not None],mode,condition)
                    write(cache,pred);pred=read(cache)
                for decoder in ['point','peak_point']:
                    row={'name':name+'_'+decoder,'run':folder.name,'checkpoint':ck,'epoch':checkpoint['epoch'],'mode':mode,'decoder':decoder,'validation':oracle_scores(pred,manifest,decoder),'inference_and_evaluation_seconds':time.perf_counter()-start};summary.append(row)
                print(name,'F1',round(summary[-2]['validation']['selected']['joint_f1'],3),'peak',round(summary[-1]['validation']['selected']['joint_f1'],3),flush=True)
                write(out/f'{pattern.replace("*","all")}_scores.json',sorted(summary,key=lambda r:r['validation']['selected']['joint_f1'],reverse=True))
            del net;torch.cuda.empty_cache()

def controlled(candidate):
    cfg=read(candidate);out=OUT/'candidates'/cfg['name'];out.mkdir(exist_ok=True);cache=out/'controlled_predictions.json';base=read(ROOT/'Dance_Tracking_Pilot/tracks.json')['roster']
    if cache.exists():return
    net,condition,_=load_model(OUT/'runs'/cfg['run']/cfg['checkpoint']);cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));pred=[];start=time.perf_counter()
    for k in range(300):
        ok,im=cap.read();assert ok;pp=predict_aug(net,im,[o for o in base['frames'][k] if o['status']=='observed'],cfg['mode'],condition)
        if cfg['decoder']!='point':
            for o in pp.values():o['point']=o[cfg['decoder']]
        pred.append(pp)
        if k%100==0:print(cfg['name'],'controlled',k,flush=True)
    cap.release();write(cache,pred);write(out/'settings.json',cfg);write(out/'timing.json',{'seconds':time.perf_counter()-start,'frames':300,'includes':'Support inference and CPU crop preprocessing; excludes YOLO detection and rendering'})

if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;p=argparse.ArgumentParser();p.add_argument('--pattern',default='v2_*');p.add_argument('--candidate');args=p.parse_args()
    if args.candidate:controlled(args.candidate)
    else:validation(args.pattern)
