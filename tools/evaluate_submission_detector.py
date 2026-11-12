"""Evaluate the three saved YOLO detectors with one full-frame AP protocol."""
import argparse
from collections import Counter
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
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Experiment"
sys.path.insert(0, str(EXP / "Dance_Overnight/code"))
from detector_probe import candidates, merge

TRIAL = EXP / "lighting_expansion_300"
PUBLISHED = EXP / "Dance_Overnight/detector_expanded/runs/diverse_brightness/weights/best.pt"
FINAL = TRIAL / "runs/detector_300_finetune_s17/weights/best.pt"
DEV = TRIAL / "detector_dev_references.json"
RESERVED = EXP / "Dance_Overnight/data/final_confirmation/labels.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def frames(video, keys):
    capture = cv2.VideoCapture(str(video))
    try:
        result = {}
        for key in keys:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(key))
            ok, image = capture.read()
            if not ok or image.shape[:2] != (720, 1280):
                raise ValueError(f"Cannot read original source frame {key}")
            result[key] = image
        return result
    finally:
        capture.release()


def label_matches(iou, confidence, threshold=.50):
    """Record matched reference indices at a fixed confidence for error review."""
    free = set(range(iou.shape[0]))
    matches = {}
    for j, score in enumerate(confidence):
        if score < .25 or not free:
            continue
        i = max(free, key=lambda k: float(iou[k, j]))
        if iou[i, j] >= threshold:
            matches[j] = i
            free.remove(i)
    return matches, free


def score(model, images, labels, device, output, plot):
    tp, scores, case_rows = [], [], {}
    targets = 0
    by_contact = {}
    duration = 0
    iouv = SimpleNamespace(iouv=torch.linspace(.5, .95, 10))
    for key, image in images.items():
        refs = [row for row in labels[key] if row["box"] is not None and not row.get("outside")]
        start = perf_counter()
        rows = merge(candidates(model, image, 960, device), .55)
        duration += perf_counter() - start
        confidence = [row["score"] for row in rows]
        scores.extend(confidence)
        targets += len(refs)
        iou = box_iou(torch.tensor([row["box"] for row in refs]).reshape(-1, 4),
                      torch.tensor([row["box"] for row in rows]).reshape(-1, 4))
        matches = BaseValidator.match_predictions(iouv, torch.zeros(len(rows)), torch.zeros(len(refs)), iou)
        tp.extend(matches.numpy())
        linked, missed = label_matches(iou, confidence)
        for i, ref in enumerate(refs):
            counts = by_contact.setdefault(ref.get("contact", "unknown"), Counter())
            counts["reference_boxes"] += 1
            counts["matched_boxes"] += int(i not in missed)
        case_rows[key] = {"references": refs, "predictions": [row for row in rows if row["score"] >= .25],
                          "matched": linked, "missed": sorted(missed)}
    output.mkdir(parents=True, exist_ok=True)
    _, _, p, r, f1, ap, *_ = ap_per_class(np.asarray(tp).reshape(-1, 10), np.asarray(scores),
        np.zeros(len(scores)), np.zeros(targets), plot=plot, names={0: "dancer"}, save_dir=output, prefix="Box")
    matched = sum(len(case["matched"]) for case in case_rows.values())
    predictions = sum(len(case["predictions"]) for case in case_rows.values())
    return {"frames": len(images), "reference_boxes": targets, "mAP50": float(ap[:, 0].mean()),
            "mAP50_95": float(ap.mean()), "precision_at_max_f1": float(p.mean()),
            "recall_at_max_f1": float(r.mean()), "max_f1": float(f1.mean()),
            "precision_at_025_iou50": matched / max(1, predictions),
            "recall_at_025_iou50": matched / targets, "matched_at_025_iou50": matched,
            "predictions_at_025": predictions, "ms_per_frame_including_merge": 1000 * duration / len(images),
            "by_contact_at_025_iou50": by_contact, "cases": case_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic", type=Path, default=ROOT / "models/yolo26m.pt")
    parser.add_argument("--device", default="0", help="0 for RTX GPU or cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/detector")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    ref = {"development": read(DEV)["labels"], "reserved_diagnostic": read(RESERVED)}
    source = ROOT / "media/source_choreography.mp4"
    subsets = {name: frames(source, rows) for name, rows in ref.items()}
    report = {"protocol": {"class": "dancer (COCO person for generic model)", "image_size": 960,
                          "detector_confidence_floor": .06, "merge_iou": .55,
                          "ap_iou": [round(.5 + i * .05, 2) for i in range(10)],
                          "scope": "Same-video development and previously inspected reserved frames with approximate, potentially incomplete references.",
                          "device": "CPU" if args.device == "cpu" else torch.cuda.get_device_name(int(args.device))},
              "models": {}}
    for name, weights in (("generic", args.generic), ("published", PUBLISHED), ("tuned_300", FINAL)):
        model = YOLO(weights)
        # Warm the predictor so startup is excluded from the small-sample timing.
        candidates(model, next(iter(subsets["development"].values())), 960, args.device)
        with weights.open("rb") as stream:
            entry = {"sha256": file_digest(stream, "sha256").hexdigest()}
        for split, images in subsets.items():
            entry[split] = score(model, images, ref[split], args.device, args.output / name / split, True)
            metric = entry[split]
            print(f"{name} {split}: {metric['mAP50']:.3f} / {metric['mAP50_95']:.3f}", flush=True)
        report["models"][name] = entry
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
