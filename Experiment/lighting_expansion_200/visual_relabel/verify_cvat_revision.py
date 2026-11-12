"""Verify every dancer and floor guide in a downloaded CVAT revision export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

P = Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from build_dataset import EXP, HOLDOUT
sys.path.insert(0,str(EXP/'code'))
from import_cvat_review import convert


def main():
    ap=argparse.ArgumentParser();ap.add_argument('export',type=Path);ap.add_argument('--task-url',required=True)
    args=ap.parse_args();xml=P/'verified_cvat_export.xml'
    if args.export.suffix.lower()=='.zip':
        with zipfile.ZipFile(args.export) as z:
            members=[n for n in z.namelist() if Path(n).name=='annotations.xml'];assert len(members)==1
            xml.write_bytes(z.read(members[0]))
    else: xml.write_bytes(args.export.read_bytes())
    actual=convert(xml);expected=json.loads((P/'visual_labels.json').read_text())
    assert actual.keys()==expected.keys() and len(actual)==200
    assert not set(map(int,actual)).intersection(HOLDOUT)
    for f,rs in actual.items():
        src={r['id']:r for r in expected[f]};assert src.keys()=={r['id'] for r in rs},f
        for r in rs:
            q=src[r['id']]
            assert r['sample_split']=='train'
            for k in ['contact','support_type']:assert r[k]==q[k],(f,r['id'],k)
            for k in ['box','floor']:
                a,b=r[k],q[k]
                assert (a is None and b is None) or (a is not None and b is not None and max(abs(x-y) for x,y in zip(a,b))<.02),(f,r['id'],k,a,b)
    def floor_rows(path):
        out={}
        for im in ET.parse(path).findall('image'):
            lines=[]
            for s in im.findall('polyline'):
                lines.append((s.attrib['label'],tuple(float(v) for pt in s.attrib['points'].split(';') for v in pt.split(','))))
            out[Path(im.attrib['name']).name]=sorted(lines)
        return out
    a,b=floor_rows(xml),floor_rows(P/'visual_revision_annotations.xml')
    assert a.keys()==b.keys()
    for name,lines in a.items():
        assert len(lines)==len(b[name])==12,name
        for (la,pa),(lb,pb) in zip(lines,b[name]):
            assert la==lb and len(pa)==len(pb) and max(abs(x-y) for x,y in zip(pa,pb))<.02,name
    report={'status':'passed','frames':200,'dancer_boxes':sum(map(len,actual.values())),
        'support_points':sum(r['floor'] is not None for rs in actual.values() for r in rs),
        'floor_polylines':2400,'task_url':args.task_url,'coordinate_tolerance_pixels':.02,
        'export_sha256':hashlib.sha256(xml.read_bytes()).hexdigest(),
        'labels_sha256':hashlib.sha256((P/'visual_labels.json').read_bytes()).hexdigest(),
        'scope':'All frames, dancer IDs, statuses, support kinds, coordinates and floor guides compared.'}
    (P/'cvat_roundtrip_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
