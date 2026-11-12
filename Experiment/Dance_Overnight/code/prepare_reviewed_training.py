"""Convert the merged CVAT revision into a new support-training manifest."""
import argparse,json,hashlib,datetime
from pathlib import Path

def prepare(source,reviewer):
    out=Path(__file__).resolve().parents[1];expected={str(r['source_frame']):r['split'] for r in json.loads((out/'CVAT_Review/development/manifest.json').read_text())};rows=json.loads(source.read_text());assert set(rows)==set(expected),'The merged review must contain exactly the71 assigned images.'
    for frame,objects in rows.items():
        assert objects,f'Empty image{frame}'
        assert all(o['sample_split']==expected[frame] for o in objects),f'Split changed in{frame}'
        assert all((o['floor'] is not None)==(o['contact']=='visible') for o in objects),f'Point/status mismatch in{frame}'
    return {'team_review':{'source_first':0,'train':sorted(int(k) for k,v in expected.items() if v=='train'),'val':sorted(int(k) for k,v in expected.items() if v=='val'),'labels':rows,'review_provenance':{'input_file':source.name,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reviewer_note':reviewer,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'New label revision. Split membership verified. The converter cannot establish whether a human actually reviewed the images. Original frozen manifests and scores are unchanged.'}}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('review',type=Path);p.add_argument('output',type=Path);p.add_argument('--reviewer',required=True);a=p.parse_args();assert not a.output.exists(),'Use a new revision filename';data=prepare(a.review,a.reviewer);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2));print('Prepared new manifest:60 training and11 validation images. Original experiments unchanged.')
