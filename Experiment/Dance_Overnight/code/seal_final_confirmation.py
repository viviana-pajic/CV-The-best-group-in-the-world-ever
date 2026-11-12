"""Validate and seal source-only manual labels before reading candidate outputs."""
from pathlib import Path
import json,hashlib,datetime
OUT=Path(__file__).resolve().parents[1];D=OUT/'data/final_confirmation'
if __name__=='__main__':
    labels=json.loads((D/'labels.json').read_text());pre=json.loads((OUT/'final_confirmation/precommit.json').read_text());assert sorted(map(int,labels))==pre['source_frames']
    for k,rows in labels.items():
        assert len({o['id'] for o in rows})==len(rows)
        for o in rows:
            b=o['box'];assert 0<=b[0]<b[2]<=1280 and 0<=b[1]<b[3]<=720;assert (o['floor'] is not None)==(o['contact']=='visible')
            if o['floor']:assert b[0]-8<=o['floor'][0]<=b[2]+8 and b[1]-8<=o['floor'][1]<=b[3]+8,(k,o)
    protocol={'sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label_sha256':hashlib.sha256((D/'labels.json').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((OUT/'final_confirmation/frozen_selection.json').read_bytes()).hexdigest(),'scope':'Seven source-only manually reviewed confirmation images; no support points or candidate detections displayed while annotating. Generic COCO proposals corrected as needed. Image-local IDs; distinguishable bodies only. Several-pixel visual uncertainty; independent human review needed.','split':'No training/development frames inside3950..4080. This is a new labeled block within the same choreography, not an independent video. No confirmation-driven tuning allowed.','evaluation_precommit':{'phases':{'new_confirmation':pre['source_frames']}}};(D/'sealed_protocol.json').write_text(json.dumps(protocol,indent=2));print('Sealed confirmation labels:',sum(map(len,labels.values())),'bodies;',sum(o['floor'] is not None for rows in labels.values() for o in rows),'visible contacts.')
