"""Validate a reviewed CVAT image export and save a new JSON revision.

Usage: python import_cvat_review.py annotations.xml reviewed_labels.json
The frozen training manifests are never modified by this command.
"""
import argparse,json,re,xml.etree.ElementTree as ET
from pathlib import Path

def attributes(shape):return {a.attrib['name']:a.text or '' for a in shape.findall('attribute')}

def convert(source):
    result={};errors=[]
    for image in ET.parse(source).findall('image'):
        name=Path(image.attrib['name']).name;match=re.fullmatch(r'frame_(\d+)\.png',name)
        if not match:errors.append(f'Unexpected image filename: {name}');continue
        frame=str(int(match[1]));boxes={};points={}
        for shape in image:
            if shape.attrib.get('label') not in ['dancer','floor_support']:continue
            attrs=attributes(shape);ident=attrs.get('instance_id') or attrs.get('dancer_id') or shape.attrib.get('group_id')
            if not ident or not ident.isdigit() or int(ident)==0:errors.append(f'{name}: shape needs a positive instance_id');continue
            ident=int(ident)
            if shape.tag=='box' and shape.attrib['label']=='dancer':
                if ident in boxes:errors.append(f'{name}: duplicate box instance {ident}')
                boxes[ident]=(shape,attrs)
            elif shape.tag=='points' and shape.attrib['label']=='floor_support':
                if ident in points:errors.append(f'{name}: duplicate support instance {ident}')
                xy=shape.attrib['points'].split(';')
                if len(xy)!=1:errors.append(f'{name}: instance {ident} must have exactly one support point');continue
                points[ident]=[float(v) for v in xy[0].split(',')]
        rows=[]
        for ident,(shape,attrs) in boxes.items():
            box=[float(shape.attrib[a]) for a in ['xtl','ytl','xbr','ybr']];contact=attrs.get('contact');point=points.get(ident);kind=attrs.get('support_type');split=attrs.get('sample_split')
            if contact not in ['visible','hidden','uncertain','airborne']:errors.append(f'{name}: instance {ident} has invalid contact={contact}')
            if (point is not None)!=(contact=='visible'):errors.append(f'{name}: instance {ident}: visible needs a point; hidden/uncertain/airborne must have none')
            if point is not None and kind not in ['feet','knees','mixed_knee_foot','seat','palms']:errors.append(f'{name}: instance {ident} needs a visible support_type')
            if not (0<=box[0]<box[2]<=int(image.attrib['width']) and 0<=box[1]<box[3]<=int(image.attrib['height'])):errors.append(f'{name}: instance {ident} invalid box')
            if point is not None and not (0<=point[0]<int(image.attrib['width']) and 0<=point[1]<int(image.attrib['height'])):errors.append(f'{name}: instance {ident} point outside image')
            if split not in ['train','val','reserved_test']:errors.append(f'{name}: instance {ident} missing/invalid sample_split')
            rows.append({'id':ident,'box':box,'floor':point,'contact':contact,'support_type':kind if point is not None and kind not in ['','none',None] else None,'sample_split':split,'outside':False,'reviewed':True,'provenance':'Imported from reviewed CVAT image export; reviewer identity must be recorded separately.'})
        if set(points)-set(boxes):errors.append(f'{name}: support points without matching boxes: {sorted(set(points)-set(boxes))}')
        result[frame]=rows
    if errors:raise ValueError('\n'.join(errors))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('xml',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new revision filename; existing labels are preserved.')
    result=convert(args.xml);args.output.write_text(json.dumps(result,indent=2));print('Validated',len(result),'images and',sum(map(len,result.values())),'dancer instances')
