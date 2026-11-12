"""Compile fresh visual contact labels and auditable proposal removals."""
import json,re,copy,hashlib,argparse
from pathlib import Path
from collections import Counter
import cv2
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight'
parser=argparse.ArgumentParser();parser.add_argument('--folder',default='expansion');parser.add_argument('--base',default='development_v1.json');parser.add_argument('--output',default='development_v2.json');parser.add_argument('--standalone',action='store_true');args=parser.parse_args();D=OUT/'data'/args.folder
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf-8')
blank=read(D/'blank.json');edits=read(D/'box_edits.json');labels={};removed=[]
for line in (D/'manual_points.txt').read_text().splitlines():
    if not line or line.startswith('#'):continue
    k,values=line.split(':');tokens=values.split();assert len(tokens)==len(blank[k]),(k,len(tokens),len(blank[k]))
    assert k not in labels,k
    labels[k]=[]
    for original,t in zip(blank[k],tokens):
        o=copy.deepcopy(original)
        if t=='D':removed.append({'frame':int(k),'proposal':o,'reason':'Duplicate, merged or invalid proposal removed during visual review'});continue
        o['box']=edits.get(k,{}).get(str(o['id']),o['box']);o['reviewed']=True;o['provenance']='Assistant visual estimate from original frame and magnified crop; not independent human ground truth'
        if t in ['H','U','A']:o.update(floor=None,contact={'H':'hidden','U':'uncertain','A':'airborne'}[t],support_type=None)
        else:
            m=re.fullmatch(r'([FKSPM])(\d+),(\d+)',t);assert m,(k,t)
            o.update(floor=[int(m[2]),int(m[3])],contact='visible',support_type={'F':'feet','K':'knees','S':'seat','P':'palms','M':'mixed_knee_foot'}[m[1]])
        labels[k].append(o)
assert set(labels)==set(blank),(set(blank)-set(labels))
if (D/'extra_labels.json').exists():
    for k,rows in read(D/'extra_labels.json').items():
        labels[k].extend(rows)
for k,rows in labels.items():
    assert len({o['id'] for o in rows})==len(rows),k
write(D/'labels.json',labels);write(D/'removed_proposals.json',removed)
manifest=read(D/'manifest.json');v2={} if args.standalone else read(OUT/'data'/args.base);assert args.folder not in v2;v2[args.folder]={'source_first':0,'labels':labels,'train':manifest['train'],'val':manifest['val']}
if not args.standalone:write(OUT/'data'/args.output,v2)
counts={s:dict(Counter(o['support_type'] if o['floor'] else o['contact'] for clip,d in v2.items() for k in d[s] for o in d['labels'][str(k)] if o['box'])) for s in ['train','val']}
write(D/'review_provenance.json',{'counts':counts,'added_frames':len(labels),'retained_proposals':sum(map(len,labels.values())),'removed_proposals':len(removed),'label_sha256':hashlib.sha256((D/'labels.json').read_bytes()).hexdigest(),'scope':'Support-crop training. Box proposals reviewed for target identity and duplicates; not an exhaustive person-detection ground-truth set. Image-local IDs are not trajectories.','definition':'Feet: midpoint of visibly planted feet or sole planted foot. Knees: midpoint of clearly supporting knees, used instead of hands when also kneeling. Seat: visible seated hip-floor contact. Palms: midpoint of clearly supporting palms when hands alone support the dancer. Hidden: supporting region occluded. Airborne: clearly no floor contact. Uncertain: ambiguous pose/contact or unclear support; exclude from supervised visibility loss in clean-label ablation. No coordinate for hidden/airborne/uncertain.','accuracy':'Approximate assistant visual labels, typically several source pixels of uncertainty. Teammate review still required.'})
(D/'audit').mkdir(exist_ok=True)
for k,rows in labels.items():
    im=cv2.imread(str(D/'images'/f'{k}.png'))
    for o in rows:
        x1,y1,x2,y2=map(round,o['box']);color=(60,230,100) if o['floor'] else (90,160,230);cv2.rectangle(im,(x1,y1),(x2,y2),color,1)
        text=f"{o['id']} {o['support_type'] or o['contact']}";cv2.putText(im,text,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        if o['floor']:cv2.drawMarker(im,tuple(o['floor']),(255,255,255),cv2.MARKER_CROSS,11,2)
    cv2.imwrite(str(D/'audit'/f'{k}.jpg'),im)
print(json.dumps(counts,indent=2));print('Compiled',len(labels),'frames;',sum(map(len,labels.values())),'reviewed crops;',len(removed),'removed proposals')
