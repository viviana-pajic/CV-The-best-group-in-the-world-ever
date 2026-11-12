"""Limited source-only identity continuation through the held-pose block.

This does not claim identity ground truth in the later crowded transition.
"""
import json,copy,datetime,hashlib
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'data/holdout';labels=json.loads((D/'labels.json').read_text())
maps={
4800:{1:3,2:1,3:2,4:4,5:5,6:6,8:7,10:8,11:9,12:10,13:11,14:12,15:14,16:13,17:15,18:16,20:17,21:18,22:20,24:19},
4835:{1:3,2:1,3:2,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:14,14:13,15:15,16:16,18:17,19:18,20:20,22:19},
4870:{1:3,2:1,3:2,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:14,14:13,15:15,16:16,17:17,18:18,19:20,101:19},
4905:{1:3,2:1,3:2,4:4,5:5,6:6,7:7,8:8,9:9,11:11,12:10,14:14,15:13,16:12,17:15,18:16,19:17,20:18,21:20,22:19},
}
result={}
for k,mapping in maps.items():
    assert set(mapping.values())==set(range(1,21));assert set(mapping)=={o['id'] for o in labels[str(k)]}
    rows=copy.deepcopy(labels[str(k)])
    for o in rows:o['image_local_id']=o['id'];o['id']=mapping[o['id']]
    result[str(k-4420)]=rows
(D/'limited_identity_labels.json').write_text(json.dumps(result,indent=2))
scope={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256((D/'limited_identity_labels.json').read_bytes()).hexdigest(),'source_frames':list(maps),'references':80,'scope':'Supplementary identity audit on the held-pose block only. Assistant visual continuation from reviewed source4719 identities through source-only bridge samples4720..4795 and the four reserved held-pose images. Nineteen dancers remain in their places; the standing soloist is distinguishable. No tracker/model predictions used. Not an independent human audit and not identity ground truth for the subsequent crowded transition.','selection':'Evaluation only; never used for model, tracker or filter selection. Primary all-frame evaluation remains image-local box matching.'}
(D/'limited_identity_protocol.json').write_text(json.dumps(scope,indent=2));print('Sealed80 held-pose identity references; crowd identities remain unclaimed')
