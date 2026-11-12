"""Compile the visually reviewed subset; ambiguous prepared frames remain unused."""
from pathlib import Path
import json,cv2,hashlib
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'Dance_Overnight/data/detector_expansion'
read=lambda p:json.loads(p.read_text())
TRAIN=[3290,3507,3857,4357];VAL=[3640]
EDITS={
 '3290':{'remove':[10],'boxes':{6:[364,364,426,434],13:[553,377,650,535]},'add':[[365,403,426,491],[791,400,835,463],[850,438,909,498]]},
 '3507':{'remove':[],'boxes':{},'add':[]},
 '3857':{'remove':[],'boxes':{},'add':[[471,392,506,455]]},
 '4357':{'remove':[],'boxes':{},'add':[[405,347,456,456]]},
 '3640':{'remove':[],'boxes':{9:[527,363,573,451],14:[884,336,919,436]},'add':[[865,349,901,435],[557,366,602,447]]},
}
if __name__=='__main__':
    blank=read(D/'blank.json');labels={};(D/'audit').mkdir(exist_ok=True)
    for k,ed in EDITS.items():
        rows=[{'id':o['id'],'box':ed['boxes'].get(o['id'],o['box']),'outside':False} for o in blank[k] if o['id'] not in ed['remove']]
        rows.extend({'id':101+i,'box':b,'outside':False} for i,b in enumerate(ed['add']));labels[k]=rows
        im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{int(k):06}.png'))
        for o in rows:
            x1,y1,x2,y2=map(round,o['box']);assert 0<=x1<x2<=1280 and 0<=y1<y2<=720
            cv2.rectangle(im,(x1,y1),(x2,y2),(80,230,120),1);cv2.putText(im,str(o['id']),(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        cv2.imwrite(str(D/'audit'/f'{k}.jpg'),im)
    (D/'reviewed_labels.json').write_text(json.dumps(labels,indent=2));(D/'reviewed_manifest.json').write_text(json.dumps({'train':TRAIN,'val':VAL,'counts':{k:len(v) for k,v in labels.items()},'scope':'Assistant visual completeness review of clearly distinguishable people in five selected images. Other prepared crowded/ending frames were not accepted as complete detector annotations. No invented boxes for fully hidden bodies.','provenance':'Reviewed original images and magnified source crops; approximate visible-extent boxes, not independent human ground truth.','edits':EDITS,'sha256':hashlib.sha256((D/'reviewed_labels.json').read_bytes()).hexdigest()},indent=2));print({k:len(v) for k,v in labels.items()})
