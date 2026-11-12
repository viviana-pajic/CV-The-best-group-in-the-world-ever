"""Prepare complete-box review, separate from support-crop proposals."""
from pathlib import Path
import json,cv2
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight/data/detector_expansion'
TRAIN=[3290,3507,3857,4157,4357,5315,5357,5400,5440,5480]
VAL=[3640,3682,3710,5510,5550]
if __name__=='__main__':
    OUT.mkdir(exist_ok=True);(OUT/'review').mkdir(exist_ok=True);data=json.loads((OUT.parent/'development_ending.json').read_text());source={d['source_first']+int(k):rows for d in data.values() for k,rows in d['labels'].items()};blank={}
    for frame in TRAIN+VAL:
        rows=[{'id':o['id'],'box':o['box']} for o in source[frame] if o['box']];blank[str(frame)]=rows;im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f'frame_{frame:06}.png'))
        for o in rows:
            x1,y1,x2,y2=map(round,o['box']);cv2.rectangle(im,(x1,y1),(x2,y2),(80,220,90),1);cv2.rectangle(im,(x1,y1-18),(x1+25,y1),(15,15,15),-1);cv2.putText(im,str(o['id']),(x1+2,y1-3),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.imwrite(str(OUT/'review'/f'{frame}.jpg'),im)
    (OUT/'blank.json').write_text(json.dumps(blank,indent=2));(OUT/'manifest.json').write_text(json.dumps({'train':TRAIN,'val':VAL,'status':'Requires visual completeness review; existing support crops alone are not detector ground truth.'},indent=2))
    print({k:len(v) for k,v in blank.items()})
