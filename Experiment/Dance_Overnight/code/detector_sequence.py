"""Cache detector candidates for reproducible development tracking experiments."""
import argparse,time
from detector_probe import *
from tracker_lab import color_feature

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--device',default='0');p.add_argument('--imgsz',type=int,default=960);a=p.parse_args();out=OUT/'sequences'/f'{a.model}_{a.imgsz}';out.mkdir(parents=True,exist_ok=True);path=out/'raw.json';weights=OUT/'runs'/a.model/'weights/best.pt';assert (weights.parents[1]/'complete.json').exists();model=YOLO(weights);cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'));pred={};ff={'seeds':[],'frames':{}};seeds=read(ROOT/'Dance_Corrected_Pilot/labels.json')['0'];start=time.perf_counter()
    if path.exists():pred=read(path)
    for k in range(300):
        ok,im=cap.read();assert ok
        if str(k) not in pred:pred[str(k)]=candidates(model,im,a.imgsz,a.device)
        ff['frames'][str(k)]=[color_feature(im,o['box']).tolist() for o in pred[str(k)]]
        if not k:ff['seeds']=[color_feature(im,o['box']).tolist() for o in seeds]
        if k%100==0:print(a.model,'development detection',k,flush=True);write(path,pred)
    cap.release();write(path,pred);write(out/'color_features.json',ff);write(out/'timing.json',{'frames':300,'seconds':time.perf_counter()-start,'imgsz':a.imgsz,'device':a.device,'scope':'Includes decoding, detector inference and appearance features. Concurrent training may affect runtime; not a standalone speed benchmark.'})

if __name__=='__main__':main()
