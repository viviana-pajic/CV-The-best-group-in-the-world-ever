"""Compare track continuity with identical detections on the bundled sample."""
from collections import defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_external_replay import DETECTOR, detect


def summarize(history, count):
    lengths = [len(rows) for rows in history.values()]
    present = {k for rows in history.values() for k, _ in rows}
    gaps, acceleration = 0, []
    for rows in history.values():
        times = np.asarray([frame for frame, _ in rows])
        centers = np.asarray([center for _, center in rows])
        gaps += int(np.sum(np.diff(times) > 1))
        if len(times) >= 3:
            consecutive = (np.diff(times) == 1)[:-1] & (np.diff(times) == 1)[1:]
            acceleration.extend((np.linalg.norm(np.diff(centers, n=2, axis=0), axis=1)[consecutive] / 12.8).tolist())
    return {"track_ids": len(lengths), "tracked_boxes": sum(lengths), "frames_with_tracks": len(present),
            "frames": count, "median_frames_per_id": float(np.median(lengths)), "within_id_gaps": gaps,
            "box_center_acceleration_p99_pct_width": float(np.quantile(acceleration, .99))}


def main():
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(DETECTOR)
    capture = cv2.VideoCapture(str(ROOT / "media/sample_static.mp4"))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    options = dict(track_high_thresh=.25, track_low_thresh=.06, new_track_thresh=.25, track_buffer=25,
                   match_thresh=.8, fuse_score=True, gmc_method="sparseOptFlow",
                   proximity_thresh=.5, appearance_thresh=.8, with_reid=False, model="auto")
    trackers = {"ByteTrack": BYTETracker(SimpleNamespace(**options)),
                "BoT-SORT": BOTSORT(SimpleNamespace(**options))}
    histories = {name: defaultdict(list) for name in trackers}
    try:
        for start in range(0, count, 8):
            images = []
            for _ in range(min(8, count - start)):
                ok, image = capture.read()
                if not ok:
                    raise ValueError("Could not decode the sample")
                images.append(image)
            for offset, (image, rows) in enumerate(zip(images, detect(model, images, device), strict=True)):
                boxes = np.asarray([r["box"] + [r["score"], 0] for r in rows], dtype=np.float32).reshape(-1, 6)
                for name, tracker in trackers.items():
                    for row in tracker.update(Boxes(boxes, image.shape[:2]), image):
                        index = int(row[-1])
                        center = (boxes[index,:2] + boxes[index,2:4]) / 2
                        histories[name][int(row[4])].append((start+offset, center))
    finally:
        capture.release()
    result = {"scope": "One 12-second same-video sample with identical merged detections. No cross-frame reference IDs; gaps and acceleration describe continuity only.",
              "config": options, "results": {name: summarize(history, count) for name, history in histories.items()}}
    output = ROOT / "experiments/tracking"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
