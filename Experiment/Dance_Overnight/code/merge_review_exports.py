"""Merge disjoint validated teammate exports, preserving train/validation membership."""
import argparse,json
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('inputs',nargs='+',type=Path);a=p.parse_args();assert not a.output.exists(),'Choose a new revision filename';master=Path(__file__).resolve().parents[1]/'CVAT_Review/development/manifest.json';expected={str(r['source_frame']):r['split'] for r in json.loads(master.read_text())};combined={}
    for path in a.inputs:
        data=json.loads(path.read_text())
        for frame,rows in data.items():
            assert frame not in combined,f'Duplicate frame{frame}; resolve conflicting reviews explicitly';assert frame in expected,f'Unexpected frame{frame}';assert rows,f'Empty reviewed image{frame}'
            assert all(o['sample_split']==expected[frame] for o in rows),f'Changed split in frame{frame}';combined[frame]=rows
    assert set(combined)==set(expected),f'Missing frames: {sorted(set(expected)-set(combined),key=int)}'
    a.output.write_text(json.dumps(dict(sorted(combined.items(),key=lambda x:int(x[0]))),indent=2));print('Merged',len(combined),'disjoint frames; original split membership preserved.')
