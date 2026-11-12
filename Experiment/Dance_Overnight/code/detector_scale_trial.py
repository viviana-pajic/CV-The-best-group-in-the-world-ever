"""Check a larger pretrained detector and more pixels before adding more labels."""
from detector_probe import *

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);dest=OUT/'scale_trial';dest.mkdir(exist_ok=True);labels=read(ROOT/'Dance_Support_Training_Pilot/labels.json');summary=[]
    models=[('frozen_large_input',OUT/'runs/posture_frozen/weights/best.pt',[1536,1920]),('coco_x',OUT/'yolo26x.pt',[960,1280])]
    for name,weights,sizes in models:
        model=YOLO(weights)
        for size in sizes:
            cache=dest/f'{name}_{size}_raw.json';raw={}
            if cache.exists():raw=read(cache)
            else:
                cap=cv2.VideoCapture(str(ROOT/'Dance_Controlled_Test/source.mp4'))
                for k in VAL:cap.set(cv2.CAP_PROP_POS_FRAMES,k);ok,im=cap.read();assert ok;raw[str(k)]=candidates(model,im,size,'0')
                cap.release();write(cache,raw)
            for nms in [.55,.65,.75]:
                pp={k:merge(r,nms) for k,r in raw.items()}
                for confidence in [.1,.15,.25,.35,.45,.55]:summary.append({'model':name,'weights':str(weights),'imgsz':size,'merge_iou':nms,'confidence':confidence,'validation':score(pp,labels,confidence)})
            print('Detector scale',name,size,flush=True)
        del model;torch.cuda.empty_cache()
    summary.sort(key=lambda r:(r['validation']['f1'],r['validation']['matched75']),reverse=True);write(dest/'results.json',summary);print('BEST SCALE',summary[0],flush=True)

if __name__=='__main__':main()
