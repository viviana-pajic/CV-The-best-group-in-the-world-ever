"""Compile fresh manual picks; never manufacture hidden coordinates."""
import json, hashlib, zipfile
from collections import Counter
import xml.etree.ElementTree as ET
import cv2
from support_label_setup import OUT,TRAIN,VAL,TEST,FRAMES

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
    labels=json.loads((OUT/'blank.json').read_text());seen=[]
    for line in (OUT/'manual_points.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        key,tokens=line.split(':');k=int(key);tokens=tokens.split();assert len(tokens)==20,(k,len(tokens));seen.append(k)
        for o,token in zip(labels[key],tokens):
            o['reviewed']=True
            if token[0] in 'FK':
                assert not o['outside'];o.update(floor=list(map(int,token[1:].split(','))),contact='visible',support_type='feet' if token[0]=='F' else 'knees')
            else:
                assert token in ['H','U','X'];assert (token=='X')==o['outside'],(k,o['id'],token)
                o.update(contact={'H':'hidden','U':'uncertain','X':'person_hidden'}[token])
    assert sorted(seen)==sorted(FRAMES)
    save('labels.json',labels)
    stats={s:dict(Counter(o['contact'] for k in ks for o in labels[str(k)])) for s,ks in [('train',TRAIN),('validation',VAL),('test',TEST)]}
    save('label_counts.json',stats)
    (OUT/'audit').mkdir(exist_ok=True)
    root=ET.Element('annotations');ET.SubElement(root,'version').text='1.1'
    meta=ET.SubElement(root,'meta');task=ET.SubElement(meta,'task');ET.SubElement(task,'name').text='Dance support pilot - assistant-reviewed labels'
    ll=ET.SubElement(task,'labels')
    for name,typ in [('dancer','rectangle'),('floor_support','points')]:
        label=ET.SubElement(ll,'label');ET.SubElement(label,'name').text=name;ET.SubElement(label,'type').text=typ;ET.SubElement(label,'color').text='#80cc80';attrs=ET.SubElement(label,'attributes')
        for an in ['dancer_id','contact','support_type']:
            at=ET.SubElement(attrs,'attribute');ET.SubElement(at,'name').text=an;ET.SubElement(at,'mutable').text='True';ET.SubElement(at,'input_type').text='text';ET.SubElement(at,'default_value').text='';ET.SubElement(at,'values').text=''
    for idx,k in enumerate(FRAMES):
        im=cv2.imread(str(OUT/'images'/f'{k:03}.png'));node=ET.SubElement(root,'image',id=str(idx),name=f'{k:03}.png',width='1280',height='720')
        for o in labels[str(k)]:
            if o['box'] is None:continue
            x1,y1,x2,y2=map(round,o['box']);co=(100,220,100) if o['floor'] else (100,170,240)
            cv2.rectangle(im,(x1,y1),(x2,y2),co,1);cv2.putText(im,f"{o['id']} {o['contact'][0].upper()}",(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,.38,co,1)
            bn=ET.SubElement(node,'box',label='dancer',source='manual',occluded='0',xtl=str(x1),ytl=str(y1),xbr=str(x2),ybr=str(y2),z_order='0',group_id=str(o['id']))
            for an,value in [('dancer_id',o['id']),('contact',o['contact']),('support_type',o['support_type'] or 'unknown')]:ET.SubElement(bn,'attribute',name=an).text=str(value)
            if o['floor']:
                p=o['floor'];cv2.drawMarker(im,tuple(p),(255,255,255),cv2.MARKER_CROSS,12,1);cv2.circle(im,tuple(p),6,(70,230,230),1)
                pn=ET.SubElement(node,'points',label='floor_support',source='manual',occluded='0',points=f'{p[0]},{p[1]}',z_order='1',group_id=str(o['id']))
                for an,value in [('dancer_id',o['id']),('contact','visible'),('support_type',o['support_type'])]:ET.SubElement(pn,'attribute',name=an).text=str(value)
        cv2.imwrite(str(OUT/'audit'/f'{k:03}.jpg'),cv2.resize(im[310:590,150:1170],(2040,560)))
    ET.indent(root);ET.ElementTree(root).write(OUT/'annotations.xml',encoding='utf-8',xml_declaration=True)
    with zipfile.ZipFile(OUT/'CVAT_images.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((OUT/'images').glob('*.png')):z.write(p,p.name)
    save('label_freeze.json',{'sha256':hashlib.sha256((OUT/'labels.json').read_bytes()).hexdigest(),'annotation_source':'Fresh assistant visual picks, approximate, no model-generated point labels. Not independently human-audited.','test_labels_fixed_before_training':True})
    print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
