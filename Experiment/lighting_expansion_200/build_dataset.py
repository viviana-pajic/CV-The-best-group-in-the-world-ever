"""Build a CVAT draft from the published baseline and lighting-diverse frames."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
EXP = ROOT / 'Experiment/Dance_Overnight'
REVIEW = EXP / 'CVAT_Review'
FIRST, LAST = 3250, 5673
HOLDOUT = range(3938, 4093)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def video_frames():
    cap = cv2.VideoCapture(str(ROOT / 'media/source_choreography.mp4'))
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES, FIRST)
    try:
        for frame in range(FIRST, LAST + 1):
            ok, image = cap.read()
            assert ok and image.shape[:2] == (720, 1280), frame
            yield frame, image
    finally:
        cap.release()


def choose_frames(existing, val, tracks):
    metrics = {}
    for frame, image in video_frames():
        if frame in HOLDOUT or frame in existing or any(abs(frame-v) <= 10 for v in val):
            continue
        observed = sum(row['status'] == 'observed' for row in tracks[frame-FIRST])
        if not observed:
            continue
        small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        metrics[frame] = [
            float(np.median(luma[15:40, 16:144])),
            float(np.median(luma[50:68, 22:138])),
            float(np.median(hsv[15:40, 16:144, 1])),
            observed,
        ]
    frames = np.array(list(metrics))
    values = np.array(list(metrics.values()))
    med = np.median(values, axis=0)
    scale = np.maximum(np.percentile(values, 90, axis=0)-np.percentile(values, 10, axis=0), 1)
    features = np.column_stack(((frames-FIRST)/240, np.clip((values-med)/scale, -2, 2)))
    distance = np.full(len(frames), np.inf)
    available = np.ones(len(frames), dtype=bool)
    chosen = []
    segments = np.array_split(np.arange(FIRST, LAST + 1), 14)
    for round_number in range(10):
        for segment in segments:
            valid = np.flatnonzero(available & (frames >= segment[0]) & (frames <= segment[-1]))
            if len(valid) == 0:
                continue
            best = valid[np.argmax(distance[valid])] if chosen else valid[len(valid)//2]
            chosen.append(best)
            distance = np.minimum(distance, ((features-features[best])**2).sum(axis=1))
            available &= np.abs(frames-frames[best]) >= 7
    while len(chosen) < 140:
        valid = np.flatnonzero(available)
        best = valid[np.argmax(distance[valid])]
        chosen.append(best)
        distance = np.minimum(distance, ((features-features[best])**2).sum(axis=1))
        available &= np.abs(frames-frames[best]) >= 7
    return {int(frames[i]): metrics[int(frames[i])] for i in sorted(chosen)}


def draft_rows(tracks, predictions):
    result = []
    for track in tracks:
        if track['status'] != 'observed':
            continue
        ident = track['id']
        x1, y1, x2, y2 = track['box']
        x1, x2 = np.clip([x1, x2], 0, 1280).tolist()
        y1, y2 = np.clip([y1, y2], 0, 720).tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        pred = predictions.get(str(ident), {})
        point = pred.get('point')
        score = pred.get('reportability', 0)
        # Seed the only support type we can infer conservatively from geometry.
        # Keep other postures and ambiguous predictions as uncertain for review.
        feet = (
            point is not None and score >= .90 and track.get('score', 0) >= .30
            and pred['augmentation_rms_pixels'] <= 4
            and (y2-y1) >= 1.3*(x2-x1)
            and x1-3 <= point[0] <= x2+3 and y2-.20*(y2-y1) <= point[1] <= y2+4
            and 0 <= point[0] < 1280 and 0 <= point[1] < 720
        )
        result.append({
            'id': ident, 'box': [x1, y1, x2, y2], 'floor': point if feet else None,
            'contact': 'visible' if feet else 'uncertain', 'support_type': 'feet' if feet else None,
            'sample_split': 'train', 'outside': False, 'reviewed': False,
            'provenance': 'Frozen tracker and support-head proposal; no human verification.',
        })
    return result


def attr(shape, name, value):
    ET.SubElement(shape, 'attribute', name=name).text = str(value)


def export_xml(labels):
    template = ET.parse(REVIEW / 'teammate_A/annotations.xml').getroot()
    root = ET.Element('annotations')
    for tag in ['version', 'meta']:
        root.append(deepcopy(template.find(tag)))
    task = root.find('meta/task')
    for tag, value in [('name', 'Lighting expansion 200 | assisted draft'), ('size', len(labels)),
                       ('start_frame', 0), ('stop_frame', len(labels)-1)]:
        task.find(tag).text = str(value)
    segments = task.find('segments')
    if segments is not None:
        task.remove(segments)
    for index, (frame, rows) in enumerate(sorted(labels.items(), key=lambda item: int(item[0]))):
        image = ET.SubElement(root, 'image', id=str(index), name=f'frame_{int(frame):06}.png',
                              width='1280', height='720')
        for row in rows:
            coords = dict(zip(['xtl', 'ytl', 'xbr', 'ybr'], (f'{v:.6f}' for v in row['box'])))
            box = ET.SubElement(image, 'box', label='dancer', occluded='0', source='auto',
                                z_order='0', **coords)
            for name, value in [('instance_id', row['id']), ('contact', row['contact']),
                                ('support_type', row['support_type'] or 'none'), ('sample_split', 'train')]:
                attr(box, name, value)
            if row['floor'] is not None:
                point = ET.SubElement(image, 'points', label='floor_support', occluded='0',
                                      source='auto', z_order='0',
                                      points=','.join(f'{v:.6f}' for v in row['floor']))
                attr(point, 'instance_id', row['id'])
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(DEST / 'annotations.xml', encoding='utf-8', xml_declaration=True)


def main():
    if (DEST / 'dataset.json').exists():
        raise FileExistsError('Use a new dataset revision folder')
    dev = read(EXP / 'data/development_ending.json')
    existing = {d['source_first']+k for d in dev.values() for split in ('train', 'val') for k in d[split]}
    val = {d['source_first']+k for d in dev.values() for k in d['val']}
    tracks = read(EXP / 'late_full_replay/base_tracks.json')['frames']
    predictions = read(EXP / 'late_full_replay/support/predictions.json')
    original = {}
    for d in dev.values():
        for k in d['train']:
            original[str(d['source_first']+k)] = [r for r in d['labels'][str(k)] if r['box'] is not None]
    chosen = choose_frames(existing, val, tracks)
    additions = {str(k): draft_rows(tracks[k-FIRST], predictions[k-FIRST]) for k in chosen}
    assert all(additions.values()) and len(original) == 60 and len(additions) == 140, (
        len(original), len(additions), [k for k, rows in additions.items() if not rows]
    )
    assert not set(chosen).intersection(existing) and not set(chosen).intersection(HOLDOUT)
    assert len(val) == 11
    combined = original | additions
    assert len(combined) == 200
    frames_dir = DEST / 'frames'
    frames_dir.mkdir(exist_ok=False)
    with zipfile.ZipFile(DEST / 'images.zip', 'w') as archive:
        for frame, image in video_frames():
            if str(frame) not in combined:
                continue
            name = f'frame_{frame:06}.png'
            path = frames_dir / name
            if str(frame) in original:
                path.write_bytes((EXP.parent / 'Dance_Project_Shared/dataset/frames' / name).read_bytes())
            else:
                assert cv2.imwrite(str(path), image)
            archive.write(path, name)
    (DEST / 'labels_schema.json').write_bytes((REVIEW / 'labels_schema.json').read_bytes())
    export_xml(combined)
    write(DEST / 'draft_labels.json', combined)
    info = {
        'baseline_training_images': len(original), 'draft_training_images': len(additions),
        'validation_images': len(val), 'confirmation_block_excluded': [3950, 4080],
        'confirmation_guard_excluded': [HOLDOUT.start, HOLDOUT.stop-1],
        'validation_guard_frames': 10,
        'selection': 'Greedy farthest point coverage across 14 time segments using two luminance regions, color saturation and observed crowd count. Minimum 7-frame spacing.',
        'provenance': 'Original 60 use published labels. New 140 use existing detector/tracker and support-model proposals. Uncertain contacts excluded from support loss.',
        'draft_contact_counts': Counter(r['contact'] for rr in additions.values() for r in rr),
        'draft_features_by_frame': chosen,
        'validation_frames': sorted(val),
        'annotation_xml_sha256': hashlib.sha256((DEST / 'annotations.xml').read_bytes()).hexdigest(),
    }
    write(DEST / 'dataset.json', info)
    print({k:v for k,v in info.items() if k != 'draft_features_by_frame'}, flush=True)


if __name__ == '__main__':
    main()
