"""Evaluate the new support model and the combined detector/support pipeline."""
import argparse
import hashlib
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from propose import EXP, FIRST, LAST, PREVIOUS, read, video_frames, write
from train_300 import OUT, DETECTOR, DETECTOR_NAME, SUPPORT_NAME, VAL_IMAGES
from evaluate_trial import calibrate, render, replay
from evaluate_final_confirmation import score
from evaluate_detector_expanded import detection_score
from evaluate_support import write as write_tracks
from detector_probe import candidates, merge
from filter_lab import export, inputs, inside
from flow_tracker import run as run_flow
from measure_display_sweep import measure
from tracker_lab import color_feature
import frozen_holdout
import trajectory_fit

FREEZE = EXP / "final_confirmation/frozen_selection.json"
SUPPORT_RUN = OUT / "runs" / SUPPORT_NAME
DETECTOR_RUN = OUT / "runs" / DETECTOR_NAME
CURRENT_DETECTOR = DETECTOR_RUN / "weights/best.pt"
FROZEN_TRACKS = EXP / "late_full_replay/base_tracks.json"
LABELS = EXP / "data/final_confirmation/labels.json"
OLD_RESULT = PREVIOUS / "runs/lighting_200_visual_s17/evaluation/results.json"
DEMO_THRESHOLD = .30


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detector_development(frozen):
    # Reuse the earlier detector-specific set when it is available in the workspace.
    pilot = EXP.parent.parent / "Dance_Support_Training_Pilot/labels.json"
    diverse = EXP.parent.parent / "Dance_Overnight/data/detector_expansion/reviewed_labels.json"
    references = OUT / "detector_dev_references.json"
    if references.exists():
        data = read(references)
        labels, frames, scope = data["labels"], data["frames"], data["scope"]
    elif pilot.exists() and diverse.exists():
        offsets = [165, 175, 185, 195, 285, 299]
        prior = read(pilot)
        labels = {str(4420+k): prior[str(k)] for k in offsets}
        labels["3640"] = read(diverse)["3640"]
        frames = sorted(map(int, labels))
        scope = "Seven earlier detector development images. The new detector trained on the 300-image train set."
    else:
        labels = {}
        for clip in read(EXP / "data/development_ending.json").values():
            for k in clip["val"]:
                labels[str(clip["source_first"] + k)] = clip["labels"][str(k)]
        frames = sorted(map(int, labels))
        scope = "Eleven support validation images. These boxes cover the annotated bodies only."
    if not references.exists():
        write(references, {"scope": scope, "frames": frames, "labels": labels})
    params = frozen["detector"]
    high = params["config"]["high_conf"]
    overlap = params["detector_merge_iou"]
    images = {k: cv2.imread(str(VAL_IMAGES / f"frame_{k:06}.png")) for k in frames}
    assert all(image is not None for image in images.values())
    results = {"scope": scope, "imgsz": 960, "merge_iou": overlap, "confidence": high, "models": {}}
    for name, weights in [("published_detector", DETECTOR), ("detector_300", CURRENT_DETECTOR)]:
        model = YOLO(weights)
        raw = {str(k): candidates(model, images[k], 960, "0") for k in frames}
        results["models"][name] = detection_score(raw, labels, frames, overlap, high)
        del model
        torch.cuda.empty_cache()
    write(DETECTOR_RUN / "development_comparison.json", results)
    return results


def rebuilt_tracks(output, frozen):
    cached = output / "base_tracks.json"
    if cached.exists():
        return read(cached)
    params = frozen["detector"]
    seed_frame = 4420
    seeds = read(EXP.parent / "Dance_Corrected_Pilot/labels.json")["0"]
    predictions, features, images = {}, {}, []
    model = YOLO(CURRENT_DETECTOR)
    for k, image in video_frames():
        rows = merge(candidates(model, image, 960, "0"), params["detector_merge_iou"])
        predictions[k] = rows
        features[k] = [color_feature(image, row["box"]).tolist() for row in rows]
        roi = image[280:600, 100:1220]
        images.append(cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (560, 160)))
        if k == seed_frame:
            seed_features = [color_feature(image, row["box"]).tolist() for row in seeds]
        if (k-FIRST) % 300 == 0:
            print("New detector inference", k-FIRST, "/", LAST-FIRST+1, flush=True)
    del model
    torch.cuda.empty_cache()

    def track(order):
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        motion = np.empty((len(order), 160, 560, 2), dtype=np.float32)
        motion[0].fill(0)
        for i, (a, b) in enumerate(zip(order, order[1:]), 1):
            motion[i] = flow.calc(images[a-FIRST], images[b-FIRST], None)
        pp = {str(i): predictions[k] for i, k in enumerate(order)}
        ff = {"seeds": seed_features, "frames": {str(i): features[k] for i, k in enumerate(order)}}
        return run_flow(pp, seeds, ff, params["config"], motion)

    forward = track(list(range(seed_frame, LAST+1)))
    backward = track(list(range(seed_frame, FIRST-1, -1)))
    for key in ("frames", "diagnostics"):
        forward[key] = backward[key][:0:-1] + forward[key]
    write_tracks(cached, forward)
    return read(cached)


def fitted_tracks(base, predictions, threshold, settings):
    points, confidence, ids = inputs(base, predictions, threshold)
    fitted, states, ages = trajectory_fit.fit(points, confidence, settings)
    return export(base, fitted, states, ages, points, ids, settings)


def export_positions(tracks, threshold, output):
    frames = [[[row["id"], *row["map"], row["map_status"], row["map_age_frames"]]
               for row in frame if inside(row.get("map"))] for frame in tracks["frames"]]
    write(output / "map_positions.json", {"first_source_frame": FIRST, "fps": 25, "threshold": threshold,
          "bounds": {"x": [0, 100], "y": [0, 62.5]},
          "axes": "x increases left to right; y increases back to front",
          "fields": ["track_id", "x", "y", "map_status", "map_age_frames"], "frames": frames})
    return sum(map(len, frames))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", choices=["support", "detector", "joint"])
    args = parser.parse_args()
    cv2.setNumThreads(3)
    torch.set_num_threads(4)
    frozen = read(FREEZE)
    if args.pipeline == "detector":
        result = detector_development(frozen)
        print({k: {m: r[m] for m in ("matched50", "matched75", "references", "predictions", "f1")}
               for k, r in result["models"].items()}, flush=True)
        return
    output = SUPPORT_RUN / ("evaluation" if args.pipeline == "support" else "joint_evaluation")
    output.mkdir(exist_ok=True)
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = read(selection_path)
    else:
        mode = frozen["support"]["mode"]
        threshold = calibrate(SUPPORT_RUN, mode, output)
        selection = {"mode": mode, "threshold": threshold, "settings": frozen["settings"],
                     "support_sha256": sha(SUPPORT_RUN / "best_joint.pt"),
                     "detector_sha256": sha(DETECTOR if args.pipeline == "support" else CURRENT_DETECTOR),
                     "scoring_scope": "Support checkpoint and threshold selected on the 11 development validation images. Detector checkpoint selected by Ultralytics on validation. No tuning on the reserved block.",
                     "display_threshold": DEMO_THRESHOLD}
        write(selection_path, selection)
    assert selection["support_sha256"] == sha(SUPPORT_RUN / "best_joint.pt")
    assert selection["detector_sha256"] == sha(DETECTOR if args.pipeline == "support" else CURRENT_DETECTOR)
    base = read(FROZEN_TRACKS) if args.pipeline == "support" else rebuilt_tracks(output, frozen)
    predictions = replay(SUPPORT_RUN, output, base, selection)
    tracks = fitted_tracks(base, predictions, selection["threshold"], selection["settings"])
    assert sha(LABELS) == read(LABELS.parent / "sealed_protocol.json")["label_sha256"]
    frozen_holdout.D = LABELS.parent
    previous = read(OLD_RESULT)
    result = {"published_baseline": previous["baseline"], "revision_200": previous["candidate"],
              f"candidate_300_{args.pipeline}": score(tracks, read(LABELS)), "selection": selection,
              "scope": "Seven reserved same-video images, sparse assistant references, full 2,424-frame offline fit. Identity accuracy is unmeasured. The same block was used by previous runs."}
    write(output / "results.json", result)
    print("Reserved comparison", {k: v["all"] for k, v in result.items() if isinstance(v, dict) and "all" in v},
          flush=True)
    demo = fitted_tracks(base, predictions, DEMO_THRESHOLD, selection["settings"])
    dots = export_positions(demo, DEMO_THRESHOLD, output)
    metrics, _ = measure(demo)
    assert metrics["displayed_dots"] == dots and metrics["frames"] == LAST-FIRST+1
    write(output / "replay_coverage.json", {"display_threshold": DEMO_THRESHOLD, **metrics})
    pipeline = "new support, frozen detections" if args.pipeline == "support" else "new detector and support"
    render(output, demo, title=f"300-image trial | {pipeline} | threshold {DEMO_THRESHOLD:g}",
           footer="0.30 display replay. Reserved comparison uses validation-calibrated reporting.")


if __name__ == "__main__":
    main()
