"""A source-only repeat review checks annotation precision, not human agreement."""
from pathlib import Path
import json,random,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight';D=OUT/'label_repeatability'

def main():
    D.mkdir(exist_ok=True);manifest=json.loads((OUT/'data/development_ending.json').read_text());groups={}
    for clip,d in manifest.items():
        for k in d['train']:
            for o in d['labels'][str(k)]:
                kind=o['support_type'] or o['contact'];groups.setdefault(kind,[]).append({'source_frame':d['source_first']+k,'annotation':o})
    rng=random.Random(3011);selected=[]
    for kind,n in [('feet',5),('knees',5),('seat',5),('palms',2),('hidden',3)]:
        rows=groups[kind];rng.shuffle(rows);selected.extend(rows[:n])
    rng.shuffle(selected);(D/'original_references.json').write_text(json.dumps(selected,indent=2))
    for i,r in enumerate(selected):
        im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f"frame_{r['source_frame']:06}.png"));b=list(map(round,r['annotation']['box']));cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),(70,220,90),1);cv2.imwrite(str(D/f'context_{i+1}.jpg'),im)
    for page in range(4):
        board=np.full((760,1320,3),24,np.uint8)
        for j,r in enumerate(selected[page*5:page*5+5]):
            i=page*5+j;im=cv2.imread(str(ROOT/'Dance_Project_Shared/dataset/frames'/f"frame_{r['source_frame']:06}.png"));b=r['annotation']['box'];l=max(0,min(1150,round((b[0]+b[2])/2-65)));t=max(250,min(610,round(b[3]-75)));patch=im[t:t+100,l:l+130];x=(j%3)*440+44;y=(j//3)*380+42;board[y:y+300,x:x+390]=cv2.resize(patch,(390,300))
            cv2.putText(board,f"{i+1} / source {r['source_frame']}",(x,y-18),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            for xx in range((l//10+1)*10,l+130,10):
                ax=x+(xx-l)*3;cv2.line(board,(ax,y),(ax,y+300),(75,75,75),1);cv2.putText(board,str(xx),(ax-10,y-3),cv2.FONT_HERSHEY_SIMPLEX,.28,(230,230,230),1)
            for yy in range((t//10+1)*10,t+100,10):
                ay=y+(yy-t)*3;cv2.line(board,(x,ay),(x+390,ay),(75,75,75),1);cv2.putText(board,str(yy),(x-34,ay+3),cv2.FONT_HERSHEY_SIMPLEX,.3,(230,230,230),1)
            a=(max(x,round(x+(b[0]-l)*3)),max(y,round(y+(b[1]-t)*3)));c=(min(x+389,round(x+(b[2]-l)*3)),min(y+299,round(y+(b[3]-t)*3)));cv2.rectangle(board,a,c,(80,180,100),1)
        cv2.imwrite(str(D/f'source_only_{page+1}.jpg'),board)
    (D/'protocol.json').write_text(json.dumps({'scope':'Blind repeat assistant review of20 sampled training crops. Existing support points/status are omitted from sheets. Same-agent repeatability, not independent human agreement and not a model test. No training/test labels or model choices will be changed from this check.','sampling':'Seed3011;5feet,5knees,5seat,2palms,3hidden; randomly reordered before displaying. Original selection strata are not printed on sheets.'},indent=2));print('Prepared20 source-only repeat crops.')

if __name__=='__main__':main()
