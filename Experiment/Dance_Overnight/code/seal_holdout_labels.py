"""Audit and seal reserved labels before any heldout model-point inference."""
from pathlib import Path
import json,hashlib,datetime
from collections import Counter
D=Path(__file__).resolve().parents[1]/'data/holdout'
labels=json.loads((D/'labels.json').read_text());bad=[]
for k,rows in labels.items():
    for o in rows:
        b=o['box'];p=o['floor']
        assert 0<=b[0]<b[2]<=1280 and 0<=b[1]<b[3]<=720,(k,o)
        assert (p is not None)==(o['contact']=='visible'),(k,o)
        if p is not None and not (b[0]-8<=p[0]<=b[2]+8 and b[1]-8<=p[1]<=b[3]+8):bad.append([k,o['id'],p,b])
assert not bad,bad
protocol={
 'sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'label_sha256':hashlib.sha256((D/'labels.json').read_bytes()).hexdigest(),
 'split':'Reserved evaluation, never included in any development/training manifest before initial frozen evaluation.',
 'annotation':'Assistant visual estimates on source images and enlarged grids. Generic COCO boxes were reviewed, duplicate/merged proposals removed or corrected, and missed distinguishable bodies added. No model support predictions were displayed.',
 'identity':'Image-local IDs, not trajectories. This set does not establish identity-switch accuracy.',
 'completeness':'Only individually distinguishable bodies are labeled. The final crowd frames have unresolved fully occluded dancers. Do not report whole-roster detection precision/recall from this annotation set or call an unmatched prediction false solely because it has no label.',
 'target':'Center of primary lower-body floor supports: midpoint of planted feet, grounded knees, or visible seat contact; a one-knee/one-foot lunge uses the midpoint of those two supports. Hands alone are used when they alone support the dancer. Hidden and uncertain have no fabricated coordinate.',
 'mixed_contact_note':'M records the combination of a grounded knee and planted foot, as seen during heldout annotation. This is descriptive metadata only; no new model output class or training change was introduced.',
 'uncertainty':'Several source pixels of visual uncertainty; extreme blur/contact ambiguity excluded as uncertain. Labels require independent teammate review before being treated as final ground truth.',
 'evaluation_precommit':{
   'oracle_crops':'Use every labeled box as a crop, with frozen checkpoint/decoder/threshold; compare against the bottom-center of the same reviewed box. No threshold optimization on these labels.',
   'end_to_end':'One-to-one Hungarian box matching at IoU0.5 against observed or short predicted boxes. Unmatched visible references count as missing positions. Also report observed-only matches and IoU0.3 as sensitivity, never choose settings by either result.',
   'metrics':['correct fraction at1/2/3/5percent mat-width error including misses','median/p90 error among available points, with coverage','visible/hidden/uncertain output status','matched hidden contacts emitted as measured','paired continuity without claiming it measures accuracy'],
   'phases':{'held_pose':[4800,4835,4870,4905],'transition_and_crowd':[4940,4975,5010,5045,5080]},
   'geometry':'Existing homography and assumed1.6aspect ratio, unchanged. Relative units, not meters.',
   'frame_correlation':'Nine correlated frames from one11.24-second section; not independent clips or proof of generalization.'
 },
 'frames':{k:dict(Counter(o['contact'] for o in rows),total=len(rows)) for k,rows in labels.items()}
}
(D/'sealed_protocol.json').write_text(json.dumps(protocol,indent=2))
print('Sealed',sum(map(len,labels.values())),'reviewed bodies;',sum(o['floor'] is not None for rows in labels.values() for o in rows),'visible support references')
