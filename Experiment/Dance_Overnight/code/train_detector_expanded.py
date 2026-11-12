"""Detector-only ablation with four newly reviewed full-frame box sets."""
from detector_train import *
import detector_train as original
import shutil
DEST=ROOT/'Dance_Overnight/detector_expanded'

def export_expanded():
    source=OUT/'dataset';target=DEST/'dataset'
    for split in ['train','val']:
        for kind in ['images','labels']:
            folder=target/kind/split;folder.mkdir(parents=True,exist_ok=True)
            for p in (source/kind/split).iterdir():
                if p.is_file():shutil.copy2(p,folder/p.name)
    labels=read(ROOT/'Dance_Overnight/data/detector_expansion/reviewed_labels.json');manifest=read(ROOT/'Dance_Overnight/data/detector_expansion/reviewed_manifest.json')
    for split in ['train','val']:
        for frame in manifest[split]:
            im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{frame:06}.png'));assert im is not None
            for tile,(l,t,r,b) in enumerate(TILES):
                w,h=r-l,b-t;lines=[]
                for o in labels[str(frame)]:
                    a,c,d,e=o['box'];a2,c2,d2,e2=max(a,l),max(c,t),min(d,r),min(e,b)
                    if d2-a2<6 or e2-c2<8 or (d2-a2)*(e2-c2)<.1*(d-a)*(e-c):continue
                    xywh=[(a2+d2-2*l)/(2*w),(c2+e2-2*t)/(2*h),(d2-a2)/w,(e2-c2)/h];lines.append('0 '+' '.join(f'{v:.7f}' for v in xywh))
                for rep in range(2 if split=='train' else 1):
                    name=f'extra{frame}_tile{tile}_r{rep}';cv2.imwrite(str(target/'images'/split/(name+'.jpg')),im[t:b,l:r],[cv2.IMWRITE_JPEG_QUALITY,98]);(target/'labels'/split/(name+'.txt')).write_text('\n'.join(lines)+'\n')
    (target/'data.yaml').write_text(f"path: '{target.as_posix()}'\ntrain: images/train\nval: images/val\nnames: [dancer]\n")
    write(DEST/'dataset_provenance.json',{'original_unique_train_frames':26,'additional_train_frames':manifest['train'],'additional_validation_frames':manifest['val'],'unique_train_frames':30,'unique_validation_frames':7,'additional_train_boxes':sum(len(labels[str(k)]) for k in manifest['train']),'scope':'Four reviewed bending/floor-work frames added to detector training, two sampled copies per added image. One extra floor-work validation image. Other prepared ambiguous crowd frames excluded. No reserved evaluation frame used.','labels_sha256':hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest()})

def warm_train(name,dark=False):
    folder=DEST/'runs'/name
    if (folder/'complete.json').exists():return
    torch.set_num_threads(4);start=time.perf_counter();model=YOLO(OUT/'runs/posture_frozen/weights/best.pt')
    result=model.train(data=str(DEST/'dataset/data.yaml'),imgsz=960,epochs=40,batch=2,workers=0,device=0,amp=True,seed=26,deterministic=True,freeze=10,optimizer='AdamW',lr0=.00015,nbs=16,warmup_epochs=2,patience=12,mosaic=0,mixup=0,copy_paste=0,scale=.10,translate=.025,hsv_h=.005,hsv_s=.15,hsv_v=.5 if dark else .15,fliplr=.5,plots=True,project=str(DEST/'runs'),name=name,exist_ok=True,verbose=False)
    write(folder/'complete.json',{'name':name,'initialization':'Previous development-trained posture_frozen checkpoint; no reserved labels','strong_brightness_augmentation':dark,'seconds':time.perf_counter()-start,'validation':result.results_dict});print('FINISHED expanded detector',name,flush=True)

if __name__=='__main__':
    DEST.mkdir(exist_ok=True);export_expanded()
    # Sequential GPU queue: support and synthetic-occlusion experiments finish first.
    while not (ROOT/'Dance_Overnight/runs/v7_occlusion_p0.6_s17/complete.json').exists():time.sleep(5)
    original.OUT=DEST;original.train('diverse_frozen',10,.0004,60)
    warm_train('diverse_warm');warm_train('diverse_brightness',dark=True)
