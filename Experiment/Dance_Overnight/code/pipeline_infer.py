"""Apply a frozen support candidate to an explicitly specified tracking result."""
import argparse,time,hashlib
from candidate_evaluate import *

def infer(cfg,base,source,out,device='cpu',reuse=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cache=out/'predictions.json'
    if cache.exists():print('Cached',out,flush=True);return
    net,condition,_=load_model(OUT/'runs'/cfg['run']/cfg['checkpoint'],device);cap=cv2.VideoCapture(str(source));pred=[];start=time.perf_counter();reuse_rows=[];reused=0
    if reuse:
        previous=Path(reuse);oldcfg=read(previous/'settings.json')
        assert all(cfg[k]==oldcfg[k] for k in ['run','checkpoint','mode','decoder'])
        assert cfg.get('temporal_radius')==oldcfg.get('temporal_radius')
        oldbase=read(previous/'base_tracks.json');oldpred=read(previous/'predictions.json');assert len(oldbase['frames'])==len(base['frames'])
        for rows,pp in zip(oldbase['frames'],oldpred):reuse_rows.append({tuple(np.round(o['box'],4)):pp[str(o['id'])] for o in rows if o['status']=='observed' and str(o['id']) in pp})
    temporal_radius=cfg.get('temporal_radius');frame_buffer={};next_frame=0;frame_count=len(base['frames'])
    for k,rows in enumerate(base['frames']):
        if temporal_radius is not None:
            radius=int(temporal_radius)
            while next_frame<=min(frame_count-1,k+radius):
                ok,frame=cap.read();assert ok,(next_frame,source);frame_buffer[next_frame]=frame;next_frame+=1
            im=[frame_buffer[max(0,k-radius)],frame_buffer[k],frame_buffer[min(frame_count-1,k+radius)]]
            for old in list(frame_buffer):
                if old<k-radius:del frame_buffer[old]
        else:ok,im=cap.read();assert ok,(k,source)
        observed=[o for o in rows if o['status']=='observed'];pp={};needed=[]
        for o in observed:
            cached=reuse_rows[k].get(tuple(np.round(o['box'],4))) if reuse_rows else None
            if cached is not None:pp[o['id']]=dict(cached);reused+=1
            else:needed.append(o)
        pp.update(predict_aug(net,im,needed,cfg['mode'],condition))
        if cfg['decoder']!='point':
            for o in pp.values():o['point']=o[cfg['decoder']]
        pred.append(pp)
        if k%100==0:print(out.name,'support',k,flush=True)
    cap.release();write(cache,pred);write(out/'settings.json',cfg);write(out/'base_tracks.json',base);write(out/'timing.json',{'seconds':time.perf_counter()-start,'frames':len(pred),'device':device,'includes':'Crop preparation and support inference only. Detector/tracker excluded.','source':str(source),'reused_identical_box_predictions':reused,'reuse_folder':str(reuse) if reuse else None})

if __name__=='__main__':
    torch.set_num_threads(4);cv2.setNumThreads(2);p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--tracks',required=True);p.add_argument('--output',required=True);p.add_argument('--source',default=str(ROOT/'Dance_Controlled_Test/source.mp4'));p.add_argument('--device',default='cpu');p.add_argument('--reuse');a=p.parse_args();base=read(a.tracks);base=base.get('roster',base);infer(read(a.candidate),base,a.source,a.output,a.device,a.reuse)
