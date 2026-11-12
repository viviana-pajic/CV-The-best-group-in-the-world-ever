"""Describe same-assistant annotation repeatability without revising frozen labels."""
from frozen_holdout import *
import re

def main():
    d=OUT/'label_repeatability';old=read(d/'original_references.json');tokens=(d/'repeat_points.txt').read_text().split();assert len(tokens)==len(old)==20;rows=[]
    for i,(r,t) in enumerate(zip(old,tokens)):
        o=r['annotation'];p=None;kind=None
        if t in ['H','U','A']:contact={'H':'hidden','U':'uncertain','A':'airborne'}[t]
        else:
            m=re.fullmatch(r'([FKSPM])(\d+),(\d+)',t);assert m;p=[int(m[2]),int(m[3])];kind={'F':'feet','K':'knees','S':'seat','P':'palms','M':'mixed_knee_foot'}[m[1]];contact='visible'
        both=p is not None and o['floor'] is not None;rows.append({'sample':i+1,'source_frame':r['source_frame'],'instance_id':o['id'],'original_contact':o['contact'],'repeat_contact':contact,'original_kind':o['support_type'],'repeat_kind':kind,'original_point':o['floor'],'repeat_point':p,'pixel_difference':float(np.linalg.norm(np.array(p)-o['floor'])) if both else None,'stage_difference_pct_width':float(np.linalg.norm(stage(p)-stage(o['floor']))) if both else None})
    both=[r for r in rows if r['pixel_difference'] is not None];result={'scope':read(d/'protocol.json')['scope'],'samples':20,'same_contact_status':sum(r['original_contact']==r['repeat_contact'] for r in rows),'both_visible':len(both),'median_pixel_difference':float(np.median([r['pixel_difference'] for r in both])),'p90_pixel_difference':float(np.percentile([r['pixel_difference'] for r in both],90)),'median_stage_difference_pct_width':float(np.median([r['stage_difference_pct_width'] for r in both])),'p90_stage_difference_pct_width':float(np.percentile([r['stage_difference_pct_width'] for r in both],90)),'different_contact_status':[r for r in rows if r['original_contact']!=r['repeat_contact']],'rows':rows,'interpretation':'Repeat placement uncertainty is not model error or a confidence interval. The same assistant reviewed the source twice; independent teammate review remains necessary. Difficult floor poses can change both point placement and visible/uncertain status. No frozen model labels or scores were altered.'};write(d/'results.json',result);print({k:v for k,v in result.items() if k not in ['rows','different_contact_status']});print('Status disagreements:',[(r['sample'],r['original_contact'],r['repeat_contact']) for r in result['different_contact_status']],flush=True)

if __name__=='__main__':main()
