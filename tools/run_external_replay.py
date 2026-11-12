"""Run the published 300-image detector and contact model on a new 16:9 replay."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "Experiment"
sys.path.insert(0, str(EXPERIMENT / "Dance_Overnight/code"))

from candidate_evaluate import load_model, predict_aug
from detector_probe import merge

RUNS = EXPERIMENT / "lighting_expansion_300/runs"
SUPPORT = RUNS / "lighting_300_visual_s17/best_joint.pt"
DETECTOR = RUNS / "detector_300_finetune_s17/weights/best.pt"
SELECTION = SUPPORT.parent / "joint_evaluation/selection.json"
INFERENCE_SIZE = (1280, 720)
TILES = [(0, 260, 610, 650), (335, 260, 945, 650), (670, 260, 1280, 650)]


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def detect(model, images, device="0"):
    crops = [image[y1:y2, x1:x2] for image in images for x1, y1, x2, y2 in TILES]
    outputs = model.predict(crops, imgsz=960, conf=.06, classes=[0], device=device,
                            verbose=False, half=device != "cpu")
    for start in range(0, len(outputs), len(TILES)):
        rows = []
        for (left, top, right, _), result in zip(TILES, outputs[start:start + len(TILES)], strict=True):
            for detection in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, confidence = map(float, detection[:5])
                x1, x2 = x1 + left, x2 + left
                if (left > 0 and x1 < left + 5) or (right < INFERENCE_SIZE[0] and x2 > right - 5):
                    continue
                rows.append({"box": [x1, y1 + top, x2, y2 + top], "score": confidence})
        yield merge(rows, .55)


def infer(source, target, detector, support, condition, selection, device="0", progress=None):
    config = {"source_sha256": digest(source), "detector_sha256": selection["detector_sha256"],
              "support_sha256": selection["support_sha256"], "mode": selection["mode"],
              "tiles": [list(tile) for tile in TILES], "detector_half": device != "cpu", "tracker": "ByteTrack"}
    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved["config"] == config and len(saved["frames"]) == saved["count"]:
            print(f"{source.stem}: using {saved['count']} completed frames", flush=True)
            return

    capture = cv2.VideoCapture(str(source))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not count or not fps or width * 9 != height * 16:
        raise ValueError(f"Expected a readable 16:9 source: {source}")
    tracker = BYTETracker(SimpleNamespace(track_high_thresh=.25, track_low_thresh=.06,
                                           new_track_thresh=.25, track_buffer=round(fps),
                                           match_thresh=.8, fuse_score=True))
    frames = []
    batch = 8
    start = perf_counter()
    print(f"{source.stem}: starting {count} frames", flush=True)
    try:
        while len(frames) < count:
            images = []
            for _ in range(min(batch, count - len(frames))):
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Decode stopped at frame {len(frames) + len(images)} in {source}")
                images.append(cv2.resize(image, INFERENCE_SIZE, interpolation=cv2.INTER_AREA))
            for image, detections in zip(images, detect(detector, images, device), strict=True):
                boxes = np.asarray([row["box"] + [row["score"], 0] for row in detections],
                                   dtype=np.float32).reshape(-1, 6)
                tracked = tracker.update(Boxes(boxes, INFERENCE_SIZE[::-1]))
                rows = []
                for row in tracked:
                    ident, detection_index = int(row[4]), int(row[-1])
                    detection = detections[detection_index]
                    rows.append({"id": ident, "box": detection["box"], "status": "observed",
                                 "score": detection["score"]})
                contacts = predict_aug(support, image, rows, selection["mode"], condition)
                for row in rows:
                    prediction = contacts[row["id"]]
                    row["point"] = prediction["point"]
                    row["reportability"] = prediction["reportability"]
                frames.append(rows)
                if len(frames) == 40 or len(frames) % 250 == 0:
                    print(f"{source.stem}: {len(frames)}/{count} at {len(frames)/(perf_counter()-start):.1f} fps",
                          flush=True)
            if progress:
                progress(len(frames), count)
    finally:
        capture.release()
    save(target, {"config": config, "count": count, "fps": fps, "source_resolution": [width, height],
                  "inference_resolution": INFERENCE_SIZE, "frames": frames,
                  "device": "CPU" if device == "cpu" else torch.cuda.get_device_name(int(device)),
                  "seconds": round(perf_counter() - start, 2)})
    print(f"{source.stem}: saved {count} frames in {perf_counter()-start:.1f}s", flush=True)


def load_checkpoints(device):
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    for path, key in ((DETECTOR, "detector_sha256"), (SUPPORT, "support_sha256")):
        if not path.is_file() or digest(path) != selection[key]:
            raise ValueError(f"Missing or mismatched checkpoint: {path.name}")
    detector = YOLO(DETECTOR)
    support, condition, _ = load_model(SUPPORT, "cpu" if device == "cpu" else f"cuda:{device}")
    return detector, support, condition, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This replay requires CUDA")
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    detector, support, condition, selection = load_checkpoints("0")
    print(f"CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}", flush=True)
    for source in args.sources:
        infer(source, source.with_suffix(".inference.json"), detector, support, condition, selection)


if __name__ == "__main__":
    main()
