"""Seal final evaluation labels before inspecting any predicted floor points there."""
from pathlib import Path
import json,hashlib,datetime
D=Path(__file__).resolve().parents[1]/'data/backup';rows=json.loads((D/'labels.json').read_text())
for k,rr in rows.items():
    for o in rr:
        b=o['box'];assert 0<=b[0]<b[2]<=1280 and 0<=b[1]<b[3]<=720
        assert (o['floor'] is not None)==(o['contact']=='visible')
        if o['floor']:assert b[0]-8<=o['floor'][0]<=b[2]+8 and b[1]-8<=o['floor'][1]<=b[3]+8,(k,o)
protocol={'sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label_sha256':hashlib.sha256((D/'labels.json').read_bytes()).hexdigest(),'scope':'Final reserved short crowd section. Seven images, individually distinguishable bodies only; not complete20-person ground truth. All visible references are standing/crouched foot support, not a new floor-work test.','annotation':'Assistant manual visual estimates from source images and magnified grids, generic COCO box proposals, no support predictions. Ambiguous target/contact marked uncertain. Duplicates removed and one missed distinguishable dancer added. Several-pixel uncertainty; independent human review required.','identity':'Image-local, no cross-frame identity claim.','split':'Never used for training, threshold/filter/model selection. Both round1 and round2 frozen before annotation began.','comparison':'Identical frozen detector, forward tracking from twenty boxes at4420, support checkpoint, and full forward context through5673. Compare raw support, first frozen filter, robust trajectory fit and box-bottom substitution with identical visibility and robust filter.','evaluation_precommit':{'phases':{'held_pose':[5125,5145,5165,5195],'moving_crowd':[5225,5245,5265]}}}
(D/'sealed_protocol.json').write_text(json.dumps(protocol,indent=2));print('Sealed',len(rows),'images',sum(map(len,rows.values())),'bodies')
