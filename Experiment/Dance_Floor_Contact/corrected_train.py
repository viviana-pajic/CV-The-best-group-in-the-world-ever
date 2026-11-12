"""Export corrected labels and train a detector and visible-support pose model."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2,numpy as np,torch
from ultralytics import YOLO
from corrected_setup import OUT,BASE,TRAIN,VAL,TEST,TILES,read,get_image
from infer import WEIGHTS

def export():
    labels=read(OUT/'labels.json');manifest=read(OUT/'manifest.json');count={}
    for mode in ('detect','pose'):
        root=OUT/'dataset'/mode
        for split in ('train','val','test'):
            for folder in ('images','labels'):(root/folder/split).mkdir(parents=True,exist_ok=True)
        for row in manifest:
            k=row['frame'];split=row['split'];im=get_image(k)
            assert all(o['reviewed'] for o in labels[str(k)])
            for tile,(l,t,r,b) in enumerate(TILES):
                w,h=r-l,b-t;lines=[]
                for o in labels[str(k)]:
                    if o.get('outside'):continue
                    a,c,d,e=o['box'];a2,c2,d2,e2=max(a,l),max(c,t),min(d,r),min(e,b)
                    if d2-a2<6 or e2-c2<8 or (d2-a2)*(e2-c2)<.10*(d-a)*(e-c):continue
                    coords=[(a2+d2-2*l)/(2*w),(c2+e2-2*t)/(2*h),(d2-a2)/w,(e2-c2)/h]
                    if mode=='pose':
                        p=o['floor']
                        coords+=([(p[0]-l)/w,(p[1]-t)/h,2] if p and l<=p[0]<r and t<=p[1]<b else [0,0,0])
                    lines.append('0 '+' '.join(f'{v:.7f}' for v in coords))
                name=f'{k:03}_tile{tile}'
                cv2.imwrite(str(root/'images'/split/(name+'.jpg')),im[t:b,l:r],[cv2.IMWRITE_JPEG_QUALITY,98])
                (root/'labels'/split/(name+'.txt')).write_text('\n'.join(lines)+'\n')
        yaml=f"path: '{root.as_posix()}'\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [dancer]\n"
        if mode=='pose':yaml+='kpt_shape: [1, 3]\nflip_idx: [0]\nkpt_names:\n  0: [floor_support]\nkpt_oks_sigmas: [0.1]\n'
        (root/'data.yaml').write_text(yaml)
    for split,frames in [('train',TRAIN),('val',VAL),('test',TEST)]:
        count[split]={'keyframes':len(frames),'tiles':len(frames)*3,'visible_boxes':sum(not o.get('outside') for k in frames for o in labels[str(k)]),'support_points':sum(o['floor'] is not None for k in frames for o in labels[str(k)]),'fully_occluded':sum(o.get('outside',False) for k in frames for o in labels[str(k)])}
    count['labels_sha256']=hashlib.sha256((OUT/'labels.json').read_bytes()).hexdigest()
    (OUT/'dataset_summary.json').write_text(json.dumps(count,indent=2))
    # Native CVAT image annotation export, preserving identity as an attribute.
    xml=ET.Element('annotations');ET.SubElement(xml,'version').text='1.1'
    for index,row in enumerate(manifest):
        k=row['frame'];im=ET.SubElement(xml,'image',id=str(index),name=f'{k:03}.png',width='1280',height='720')
        for o in labels[str(k)]:
            if o.get('outside'):continue
            a,b,c,d=o['box'];shape=ET.SubElement(im,'box',label='dancer',xtl=str(a),ytl=str(b),xbr=str(c),ybr=str(d),occluded=str(int(o['floor'] is None)),source='manual',z_order='0')
            ET.SubElement(shape,'attribute',name='dancer_id').text=f'D{o["id"]:02}'
            ET.SubElement(shape,'attribute',name='contact').text=o['contact']
            if o['floor']:
                x,y=o['floor'];shape=ET.SubElement(im,'points',label='floor_support',points=f'{x},{y}',occluded='0',source='manual',z_order='1')
                ET.SubElement(shape,'attribute',name='dancer_id').text=f'D{o["id"]:02}'
    ET.indent(xml);ET.ElementTree(xml).write(OUT/'corrected_keyframes_cvat.xml',encoding='utf-8',xml_declaration=True)
    print(json.dumps(count,indent=2),flush=True)

def train(mode,epochs=50):
    started=time.perf_counter();model=YOLO(WEIGHTS/('yolo26m.pt' if mode=='detect' else 'yolo26m-pose.pt'))
    if mode=='detect':
        source=model.model.model[-1]
        def keep_person(trainer):
            with torch.no_grad():
                for net in (trainer.model,trainer.ema.ema):
                    for name in ('cv3','one2one_cv3'):
                        for src,dst in zip(getattr(source,name),getattr(net.model[-1],name)):
                            dst[-1].weight.copy_(src[-1].weight[:1]);dst[-1].bias.copy_(src[-1].bias[:1])
        model.add_callback('on_pretrain_routine_end',keep_person)
    result=model.train(data=str(OUT/'dataset'/mode/'data.yaml'),imgsz=960,epochs=epochs,batch=2,workers=0,
      device=0,amp=True,seed=26,deterministic=True,freeze=10,optimizer='AdamW',lr0=.0004,nbs=16,
      warmup_epochs=2,patience=12,mosaic=0,mixup=0,copy_paste=0,scale=.10,translate=.025,
      hsv_h=.005,hsv_s=.15,hsv_v=.15,fliplr=.5,plots=True,
      project=str(OUT/'runs'),name=mode,exist_ok=False,verbose=False)
    payload={'weights':str(model.trainer.best),'elapsed_seconds':time.perf_counter()-started,'validation':result.results_dict,
      'training_labels_sha256':read(OUT/'dataset_summary.json')['labels_sha256'],'epochs_requested':epochs}
    (OUT/f'training_{mode}.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['export','detect','pose']);p.add_argument('--epochs',type=int,default=50);a=p.parse_args()
    if a.mode=='export':export()
    else:train(a.mode,a.epochs)
