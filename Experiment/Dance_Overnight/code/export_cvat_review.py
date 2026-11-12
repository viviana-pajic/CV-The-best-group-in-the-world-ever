"""Portable image-task labels, with contacts linked to boxes by group and instance ID."""
from pathlib import Path
import json,zipfile,xml.etree.ElementTree as ET
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';DEST=OUT/'CVAT_Review';DEST.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2))
names=['instance_id','contact','support_type','sample_split']
schema=[{'name':name,'type':kind,'attributes':[{'name':a,'mutable':True,'input_type':'text','default_value':'0','values':['0']} for a in names]} for name,kind in [('dancer','rectangle'),('floor_support','points')]]
choices={'contact':['uncertain','visible','hidden','airborne'],'support_type':['none','feet','knees','mixed_knee_foot','seat','palms'],'sample_split':['unset','train','val','reserved_test']}
for label in schema:
    for attribute in label['attributes']:
        if attribute['name'] in choices:
            values=choices[attribute['name']];attribute.update(input_type='select',values=values,default_value=values[0])
    if label['name']=='floor_support':label['attributes']=[a for a in label['attributes'] if a['name']=='instance_id']
write(DEST/'labels_schema.json',schema)

def export(name,samples):
    folder=DEST/name;folder.mkdir(exist_ok=True);root=ET.Element('annotations');ET.SubElement(root,'version').text='1.1';meta=ET.SubElement(root,'meta');task=ET.SubElement(meta,'task');ET.SubElement(task,'name').text=name;ET.SubElement(task,'size').text=str(len(samples));ET.SubElement(task,'mode').text='annotation';ET.SubElement(task,'overlap').text='0';ET.SubElement(task,'start_frame').text='0';ET.SubElement(task,'stop_frame').text=str(len(samples)-1);ls=ET.SubElement(task,'labels')
    for label in schema:
        el=ET.SubElement(ls,'label');ET.SubElement(el,'name').text=label['name'];ET.SubElement(el,'color').text='#4cd68a' if label['name']=='floor_support' else '#e0b35e';ET.SubElement(el,'type').text=label['type'];attrs=ET.SubElement(el,'attributes')
        for a in label['attributes']:
            aa=ET.SubElement(attrs,'attribute')
            for k in ['name','input_type','default_value']:ET.SubElement(aa,k).text=a[k]
            ET.SubElement(aa,'mutable').text='true';ET.SubElement(aa,'values').text='\n'.join(a['values'])
    manifest=[]
    with zipfile.ZipFile(folder/'images.zip','w',zipfile.ZIP_STORED) as archive:
        for i,(frame,split,rows) in enumerate(sorted(samples,key=lambda x:x[0])):
            filename=f'frame_{frame:06}.png';archive.write(ROOT/'Dance_Project_Shared/dataset/frames'/filename,filename);image=ET.SubElement(root,'image',id=str(i),name=filename,width='1280',height='720');manifest.append({'cvat_image_id':i,'source_frame':frame,'filename':filename,'split':split,'instances':len(rows)})
            for o in rows:
                if o['box'] is None:continue
                x1,y1,x2,y2=o['box'];values={'instance_id':str(o['id']),'contact':o['contact'],'support_type':o['support_type'] or 'none','sample_split':split}
                box=ET.SubElement(image,'box',label='dancer',source='manual',occluded='0',xtl=str(x1),ytl=str(y1),xbr=str(x2),ybr=str(y2),z_order='0',group_id=str(o['id']))
                shapes=[box]
                if o['floor'] is not None:shapes.append(ET.SubElement(image,'points',label='floor_support',source='manual',occluded='0',points=','.join(map(str,o['floor'])),z_order='1',group_id=str(o['id'])))
                for shape in shapes:
                    for key,value in values.items():
                        if shape.tag=='box' or key=='instance_id':ET.SubElement(shape,'attribute',name=key).text=value
    ET.indent(root);ET.ElementTree(root).write(folder/'annotations.xml',encoding='utf-8',xml_declaration=True);write(folder/'manifest.json',manifest)
    parsed=ET.parse(folder/'annotations.xml');assert len(parsed.findall('image'))==len(samples)
    assert len(parsed.findall('image/points'))==sum(o['floor'] is not None for _,_,rows in samples for o in rows)
    print(name,len(samples),'images',len(parsed.findall('image/box')),'boxes',len(parsed.findall('image/points')),'points',flush=True)

review_revision=OUT/'data/development_review_revision.json'
manifest=read(review_revision if review_revision.exists() else OUT/'data/development_expanded.json');samples=[]
if (OUT/'data/ending/labels.json').exists():
    ending=read(OUT/'data/ending/labels.json');split=read(OUT/'data/ending/manifest.json');manifest['ending']={'source_first':0,'labels':ending,'train':split['train'],'val':split['val']}
for clip,d in manifest.items():
    for split in ['train','val']:
        for k in d[split]:samples.append((d['source_first']+k,split,d['labels'][str(k)]))
assert len({x[0] for x in samples})==len(samples)
export('development',samples)
training=[s for s in samples if s[1]=='train'];validation=[s for s in samples if s[1]=='val']
for i,name in enumerate('ABC'):export('teammate_'+name,training[i::3]+validation[i::3])
holdout=read(OUT/'data/holdout/labels.json');export('reserved_evaluation',[(int(k),'reserved_test',rows) for k,rows in holdout.items()])
if (OUT/'data/backup/labels.json').exists():
    backup=read(OUT/'data/backup/labels.json');export('final_reserved_evaluation',[(int(k),'reserved_test',rows) for k,rows in backup.items()])
if (OUT/'data/ending/labels.json').exists():
    ending=read(OUT/'data/ending/labels.json');split=read(OUT/'data/ending/manifest.json');export('supplemental_ending',[(int(k),'train' if int(k) in split['train'] else 'val',rows) for k,rows in ending.items()])
if (OUT/'data/final_confirmation/sealed_protocol.json').exists():
    labels=read(OUT/'data/final_confirmation/labels.json');export('late_confirmation',[(int(k),'reserved_test',rows) for k,rows in labels.items()])
(DEST/'IMPORT.txt').write_text('CVAT image-task review package\n\n1. Create an image annotation task with the two labels in labels_schema.json.\n2. Upload development/images.zip as task data.\n3. Import development/annotations.xml with CVAT for images 1.1.\n4. Review boxes, contacts and attributes. Box and point share a group_id and instance_id.\n5. Keep sample_split unchanged. Uncertain contacts have no point and must be excluded from supervised contact visibility loss. Hidden and airborne also have no point, but are negative reportability examples.\n\nThese are assistant visual estimates, not independently checked human ground truth. Image-local instance IDs do not establish tracking between images. The source training data are support-crop examples, not an exhaustive detector-label set.\n\nreserved_evaluation is a separate task and must stay separate from training. Do not combine the two ZIP files in one upload.\n',encoding='utf-8')
