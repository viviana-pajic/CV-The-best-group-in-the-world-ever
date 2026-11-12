"""Train from the CVAT export, with the published validation set and no holdout tuning."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import sys
import zipfile

from build_dataset import DEST, EXP, HOLDOUT, read, write

sys.path.insert(0, str(EXP/'code'))
from import_cvat_review import convert
import train_support


def build_manifest(source, expected_path, revision_dir, task_url):
    revision_dir.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as z:
            members = [n for n in z.namelist() if Path(n).name == 'annotations.xml']
            assert len(members) == 1
            xml = revision_dir/'cvat_export.xml'
            xml.write_bytes(z.read(members[0]))
    else:
        xml = source
    if expected_path.name == 'visual_labels.json':
        verification = read(expected_path.parent/'cvat_roundtrip_check.json')
        assert verification['status'] == 'passed'
        assert verification['export_sha256'] == hashlib.sha256(xml.read_bytes()).hexdigest()
        assert verification['labels_sha256'] == hashlib.sha256(expected_path.read_bytes()).hexdigest()
    exported = convert(xml)
    expected = read(expected_path)
    assert set(exported) == set(expected) and len(exported) == 200
    assert all(k not in HOLDOUT for k in map(int, exported))
    for frame, items in exported.items():
        original = {r['id']: r for r in expected[frame]}
        assert set(original) == {r['id'] for r in items}, frame
        assert all(r['sample_split'] == 'train' for r in items), frame
        for row in items:
            prev = original[row['id']]
            assert row['contact'] == prev['contact'] and row['support_type'] == prev['support_type']
            for key in ['box', 'floor']:
                a, b = row[key], prev[key]
                assert (a is None and b is None) or (
                    a is not None and b is not None and max(abs(x-y) for x,y in zip(a,b)) < .02
                ), (frame, row['id'], key, a, b)
            row['reviewed'] = False
            row['provenance'] = prev.get('provenance', 'Original published label')
        assert any(r['contact'] != 'uncertain' for r in items)
    val = {}
    for clip in read(EXP/'data/development_ending.json').values():
        for k in clip['val']:
            val[str(clip['source_first']+k)] = clip['labels'][str(k)]
    assert len(val) == 11 and not set(val).intersection(exported)
    root = DEST/'training_input'
    images = root/'Dance_Project_Shared/dataset/frames'
    images.mkdir(parents=True, exist_ok=True)
    for frame in exported.keys() | val.keys():
        name = f'frame_{int(frame):06}.png'
        src = (DEST/'frames' if frame in exported else
               EXP.parent/'Dance_Project_Shared/dataset/frames')/name
        if not (images/name).exists():
            os.link(src, images/name)
    data = {'lighting_200': {
        'source_first': 0, 'train': sorted(map(int, exported)), 'val': sorted(map(int, val)),
        'labels': exported | val,
        'provenance': {'cvat_xml_sha256': hashlib.sha256(xml.read_bytes()).hexdigest(),
                       'cvat_task': task_url, 'expected_labels_sha256': hashlib.sha256(expected_path.read_bytes()).hexdigest(),
                       'scope': 'Assistant visual revision; approximate source-image labels, not human verified.'
                           if expected_path.name == 'visual_labels.json' else 'AI-assisted pilot; unverified model proposals retained with provenance.'},
    }}
    path = revision_dir/'training_manifest.json'
    write(path, data)
    print('Training:', len(exported), 'frames,', Counter(
        r['contact'] for rr in exported.values() for r in rr), flush=True)
    return path, root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cvat_export', type=Path)
    parser.add_argument('--task-url', required=True)
    parser.add_argument('--name', default='lighting_200_s17')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--expected-labels', type=Path, default=DEST/'refined_labels.json')
    parser.add_argument('--revision-dir', type=Path, default=DEST)
    args = parser.parse_args()
    assert not (DEST/'runs'/args.name).exists(), 'Use a new training run name'
    manifest, stage = build_manifest(args.cvat_export, args.expected_labels, args.revision_dir, args.task_url)
    train_support.ROOT = stage
    train_support.OUT = DEST
    train_support.train(args.name, manifest, seed=17, wide=True, attention=True,
                        ignore_uncertain=True, weight_floor=30, epochs=args.epochs)
