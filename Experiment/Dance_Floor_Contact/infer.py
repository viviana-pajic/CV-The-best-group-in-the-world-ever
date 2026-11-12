"""Cache detections, COCO joints, and the learned contact proposals for this video."""

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / 'Dance_Training_Pilot'
REVIEW = ROOT.parent / 'Dance_CVAT_Pilot'
SOURCE = ROOT.parent / 'Dance_Project_Shared' / 'dataset' / 'frames'
WEIGHTS = ROOT.parent / 'Dance YOLO trial' / 'models'
CLIPS = {'reviewed': range(4420, 4495), 'turns': range(4540, 4611)}
TILES = [(0, 238, 620, 627), (345, 238, 965, 627), (660, 238, 1280, 627)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def image(frame):
    img = cv2.imread(str(SOURCE / f'frame_{frame:06}.png'))
    if img is None:
        raise FileNotFoundError(f'Frame {frame}')
    return img


def pose_rows(result, offset=(0, 0), region=None):
    boxes = result.boxes.data.cpu().numpy()
    points = result.keypoints.data.cpu().numpy()
    x, y = offset
    rows = []
    for box, point in zip(boxes, points):
        x1,y1,x2,y2 = box[:4] + [x,y,x,y]
        if region is not None:
            l,t,r,b = region
            if (l and x1 < l+4) or (r < 1280 and x2 > r-4) or y1 < t+4 or y2 > b-4:
                continue
        xy = point + np.array([x,y,0])
        rows.append({'box': [round(float(v),2) for v in (x1,y1,x2,y2)],
                     'score': round(float(box[4]),4), 'kp': xy.round(4).tolist()})
    return rows


def main():
    frames = sorted({r['source_frame'] for r in read(PILOT / 'split_manifest.json')}
                    | {i for group in CLIPS.values() for i in group})
    path = ROOT / 'predictions.json'
    cache = read(path) if path.exists() else {}
    need = [i for i in frames if str(i) not in cache]
    det = YOLO(WEIGHTS / 'yolo26m.pt')
    pose = YOLO(WEIGHTS / 'yolo26m-pose.pt')
    learned = YOLO(PILOT / read(PILOT / 'run_pose.json')['weight'])
    options = dict(device=0, classes=[0], quantize=16, verbose=False)
    for start in range(0, len(need), 4):
        batch = need[start:start+4]
        images = [image(i) for i in batch]
        detections = det.predict(images, imgsz=960, conf=.09, **options)
        poses = pose.predict(images, imgsz=960, conf=.16, **options)
        learned_points = learned.predict(images, imgsz=960, conf=.16, **options)
        for i, img, d, p, k in zip(batch, images, detections, poses, learned_points):
            tile_preds = pose.predict([img[y1:y2,x1:x2] for x1,y1,x2,y2 in TILES],
                                      imgsz=768, conf=.18, **options)
            rows = pose_rows(p)
            for region, pred in zip(TILES, tile_preds):
                rows.extend(pose_rows(pred, region[:2], region))
            cache[str(i)] = {'boxes': d.boxes.data.cpu().numpy().round(4).tolist(),
                             'pose': rows, 'learned': pose_rows(k)}
        if start % 24 == 0 or start+4 >= len(need):
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(cache,separators=(',',':')),encoding='utf-8')
            temp.replace(path)
            print(f'Cached {min(start+4,len(need))}/{len(need)} new frames', flush=True)


if __name__ == '__main__':
    main()
