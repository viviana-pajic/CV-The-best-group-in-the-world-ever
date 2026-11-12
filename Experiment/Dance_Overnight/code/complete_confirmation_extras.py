"""Manual source-only additions for missed distinguishable bodies in confirmation."""
from pathlib import Path
import json,re
D=Path(__file__).resolve().parents[1]/'data/final_confirmation'
EXTRAS={
3999:[([265,435,345,523],'H','Gray middle dancer behind the foreground white dress')],
4021:[([239,393,326,474],'H','Rear gray dancer'),([282,430,353,516],'H','Middle brown dancer'),([187,446,307,542],'F282,531','Foreground white dress with planted feet'),([309,462,457,562],'F419,540','Foreground green dancer in a supported back bend'),([638,394,692,456],'H','Rear white dancer')],
4043:[([343,460,445,563],'F411,547','Foreground green dancer, both feet planted'),([204,385,331,482],'H','Rear left gray dancer'),([259,435,338,508],'H','Middle brown dancer'),([509,462,580,534],'K527,511','Front right kneeling dancer'),([476,419,558,496],'S523,485','Middle seated brown dancer'),([472,398,562,465],'H','White dancer immediately behind seated brown dancer'),([482,388,559,443],'H','Rear center gray dancer'),([639,436,741,508],'K660,495','Right foreground kneeling dancer'),([573,411,637,480],'U','White dancer in a partly obscured side-supported pose'),([627,400,692,466],'H','Rear white dancer'),([644,393,714,449],'H','Farthest right white dancer')],
4061:[([223,405,289,474],'H','Rear left white dancer'),([245,419,305,498],'H','Middle left gray dancer'),([357,407,413,466],'H','Gray dancer below rear white body'),([262,446,330,533],'H','Brown dancer behind foreground white dress'),([636,397,711,456],'H','Rear right white dancer')],
4079:[([620,404,680,464],'H','Rear dancer above foreground kneeling person'),([295,436,350,500],'H','Middle gray dancer behind white foreground body'),([246,405,289,483],'H','Rear left brown dancer')]
}
if __name__=='__main__':
    rows=json.loads((D/'extra_labels.json').read_text())
    for k,items in EXTRAS.items():
        rr=[]
        for i,(box,token,note) in enumerate(items,101):
            point=None;kind=None
            if token in ['H','U','A']:contact={'H':'hidden','U':'uncertain','A':'airborne'}[token]
            else:
                m=re.fullmatch(r'([FKSPM])(\d+),(\d+)',token);assert m;point=[int(m[2]),int(m[3])];kind={'F':'feet','K':'knees','S':'seat','P':'palms','M':'mixed_knee_foot'}[m[1]];contact='visible'
            rr.append({'id':i,'box':box,'floor':point,'contact':contact,'support_type':kind,'outside':False,'reviewed':True,'identity_scope':'image-local; not a cross-frame ID','provenance':'Assistant source-only visual estimate; '+note+'. Independent human review required.'})
        rows[str(k)]=rr
    (D/'extra_labels.json').write_text(json.dumps(rows,indent=2));print('Added',sum(map(len,rows.values())),'manually reviewed missed-body boxes.')
