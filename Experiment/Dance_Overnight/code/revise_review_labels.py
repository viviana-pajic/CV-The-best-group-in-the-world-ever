"""Correct known annotation issues in the HUMAN REVIEW COPY only."""
from pathlib import Path
import json,copy,datetime
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'Dance_Overnight'
if __name__=='__main__':
    data=json.loads((OUT/'data/development_expanded.json').read_text());boxes=json.loads((OUT/'data/detector_expansion/reviewed_labels.json').read_text());notes=[]
    for clip,d in data.items():
        for k,original in list(d['labels'].items()):
            source=str(d['source_first']+int(k))
            if source not in boxes:continue
            old={o['id']:o for o in original};revised=[]
            for b in boxes[source]:
                ident=b['id'];o=copy.deepcopy(old.get(ident,{'id':ident,'floor':None,'contact':'uncertain','support_type':None,'outside':False,'reviewed':False,'identity_scope':'image-local; not a trajectory'}));o['box']=b['box'];o['provenance']='Post-experiment review revision; not used to train or rescore frozen models.'
                if source=='3290':
                    if ident==11:o.update(floor=old[10]['floor'],contact='visible',support_type=old[10]['support_type'])
                    if ident==6:o.update(floor=None,contact='hidden',support_type=None)
                    if ident==101:o.update(floor=old[6]['floor'],contact='visible',support_type=old[6]['support_type'])
                    if ident==102:o.update(floor=None,contact='hidden',support_type=None)
                if source=='3857' and ident==101:o.update(floor=None,contact='hidden',support_type=None)
                revised.append(o)
            d['labels'][k]=revised;notes.append({'source_frame':int(source),'before_boxes':len(original),'review_boxes':len(revised),'notes':'Applied the separately reviewed body boxes. Newly added ambiguous contacts stay uncertain with no invented point. Source3290 duplicate white dancer removed; its existing foot target linked to the full-body box. Existing brown-dancer contact transferred out of a merged box. Neighboring source frames3260..3335 checked to resolve the white dancer overlap.'})
    (OUT/'data/development_review_revision.json').write_text(json.dumps(data,indent=2));(OUT/'CVAT_Review/review_revision.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Corrections for teammate review only. Frozen training/evaluation manifests and reported scores are unchanged.','changes':notes},indent=2));print('Prepared corrected human-review revision',notes)
