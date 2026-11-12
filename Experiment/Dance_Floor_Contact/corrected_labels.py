"""Apply explicit visual edits; retain draft provenance and review status."""
import json
import numpy as np
from corrected_setup import OUT,BASE,FRAMES,read,sheets

def main():
    labels=read(OUT/'draft.json');edits=read(OUT/'visual_edits.json')
    lines=[s.split() for s in (BASE/'Dance_CVAT_Pilot/full_clip_keyframes.txt').read_text().splitlines() if s and not s.startswith('#')]
    fs=list(map(int,lines[0]));refs={}
    for row in lines[1:]:
        vals=[];prev=None
        for val in row[1:]:
            if val=='=':val=prev
            prev=val;vals.append(None if val[0] in 'hu' else list(map(int,val.split(','))))
        refs[int(row[0][1:])]=dict(zip(fs,vals))
    for key,edit in edits.items():
        k=int(key)
        for o in labels[key]:
            i=o['id'];o['reviewed']=True;o['box_source']='detector proposal, visually reviewed'
            if str(i) in edit.get('boxes',{}):o['box']=edit['boxes'][str(i)];o['box_source']='visual correction'
            if 'points' in edit:
                p=edit['points'][i-1]
                o['floor']=p;o['point_source']='visual pick' if p else 'hidden or uncertain; no coordinate'
            elif k<=74:
                a=max(f for f in fs if f<=k);b=min(f for f in fs if f>=k)
                pa,pb=refs[i][a],refs[i][b]
                o['floor']=None if pa is None or pb is None else (np.rint(np.asarray(pa) if a==b else np.asarray(pa)+(np.asarray(pb)-pa)*(k-a)/(b-a)).astype(int).tolist())
                o['point_source']='existing reviewed contacts, interpolated then visually checked on this selected frame'
            if str(i) in edit.get('floor',{}):o['floor']=edit['floor'][str(i)];o['point_source']='visual correction'
            o['contact']='visible' if o['floor'] else 'hidden_or_uncertain'
            o['outside']=i in edit.get('outside',[])
            if o['outside']:
                o['box']=None;o['floor']=None;o['contact']='person_fully_occluded';o['box_source']='no reliably visible extent; excluded from detection labels'
            elif o['floor']:
                # A reviewed support point is part of the person's extent.
                # Keep a 2px margin, recording expansion separately for audit.
                x,y=o['floor'];a,b,c,d=o['box']
                new=[min(a,x-2),min(b,y-2),max(c,x+2),max(d,y+2)]
                if new!=o['box']:o['box_before_contact_margin']=o['box'];o['box']=new
    (OUT/'labels.json').write_text(json.dumps(labels,indent=2))
    sheets([int(k) for k in edits])
    print('Reviewed keyframes',len(edits),'of',len(FRAMES))
if __name__=='__main__':main()
