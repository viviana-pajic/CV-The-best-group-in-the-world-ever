"""Replay the frozen Wicked Games detector and floor-contact evaluation."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

from build_dashboard_data import stage_projection
from evaluate_submission_detector import FINAL, PUBLISHED, ROOT, TRIAL, candidates, label_matches, merge
from compare import match_boxes


EVIDENCE = ROOT / "experiments/final_evaluation"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify_inputs():
    locked = read_json(EVIDENCE / "frozen-inputs.json")
    for path, expected in locked.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen holdout file changed: {path}")
    return locked


def detection_metrics(annotations, predictions, frames, floor):
    validator = SimpleNamespace(iouv=torch.linspace(.5, .95, 10))
    tp_rows, confidence = [], []
    n_references = 0
    cases = {}
    for frame in frames:
        references = annotations[str(frame)]["dancers"]
        rows = predictions[frame]
        n_references += len(references)
        scores = [row["score"] for row in rows]
        confidence.extend(scores)
        gt = torch.tensor([box for box, _ in references]).reshape(-1, 4)
        predicted = torch.tensor([row["box"] for row in rows]).reshape(-1, 4)
        iou = box_iou(gt, predicted)
        matched_ap = BaseValidator.match_predictions(
            validator, torch.zeros(len(rows)), torch.zeros(len(references)), iou
        ).numpy()
        tp_rows.extend(matched_ap)
        linked, missed = label_matches(iou, scores)
        confident = [row for row in rows if row["score"] >= floor]
        cases[str(frame)] = {
            "predictions": confident,
            "matches": linked,
            "missed": sorted(missed),
            "false_positives": [j for j in range(len(confident)) if j not in linked],
        }
    tp = sum(len(case["matches"]) for case in cases.values())
    fp = sum(len(case["false_positives"]) for case in cases.values())
    fn = n_references - tp
    _, _, _, _, _, ap, *_ = ap_per_class(
        np.asarray(tp_rows, dtype=bool).reshape(-1, 10),
        np.asarray(confidence), np.zeros(len(confidence)), np.zeros(n_references),
    )
    return {
        "frames": len(frames), "reference_boxes": n_references,
        "ap50": float(ap[:, 0].mean()), "ap50_95": float(ap.mean()),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / max(1, tp + fp), "recall": tp / n_references,
        "f1": 2 * tp / max(1, 2 * tp + fp + fn), "cases": cases,
    }


def project_point(matrix, point, stage):
    x, y, w = matrix @ [*point, 1]
    return np.array([x / w, y / w]) * stage


def contact_metrics(protocol, annotations, raw_path=None):
    source = ROOT / protocol["source"]
    script = source.with_suffix(".js").read_text(encoding="utf-8").strip()
    prefix = "(window.DANCE_CLIPS ??= []).push("
    if not script.startswith(prefix) or not script.endswith(");"):
        raise ValueError("Unexpected dashboard data format")
    replay = json.loads(script[len(prefix):-2])
    calibration_file = ROOT / "experiments/replays" / (source.stem + ".calibration.json")
    calibration = read_json(calibration_file)
    if replay["fps"] != protocol["fps"] or len(replay["frames"]) != len(calibration["frame_corners"]):
        raise ValueError("The replay and calibration must cover the same frames at the recorded FPS")
    image_scale = np.asarray(replay["resolution"]) / protocol["annotation"]["image_size"]
    stage = np.asarray(replay["stageMeters"])
    unit_scale = stage / replay["stage"]
    raw_cache = read_json(raw_path) if raw_path else None
    if raw_cache:
        settings = read_json(TRIAL / "runs/lighting_300_visual_s17/joint_evaluation/selection.json")
        expected = {
            "source_sha256": digest(source),
            "detector_sha256": settings["detector_sha256"],
            "support_sha256": settings["support_sha256"],
            "mode": settings["mode"],
        }
        if (any(raw_cache["config"][key] != value for key, value in expected.items())
                or raw_cache["fps"] != replay["fps"]
                or raw_cache["count"] != len(replay["frames"])):
            raise ValueError("Raw contact cache is from another run")
        raw_scale = np.asarray(raw_cache["source_resolution"]) / raw_cache["inference_resolution"]
    methods = ("box_bottom", "dashboard_solid")
    rows = []
    extra_solid = 0
    for frame in protocol["frames"]:
        boxes, points, *_ = replay["frames"][frame - replay["firstFrame"]]
        refs = annotations[str(frame)]["dancers"]
        solid = {p[0]: np.asarray(p[1:3]) * unit_scale for p in points if not p[3]}
        if raw_cache:
            raw_by_id = {row["id"]: row for row in raw_cache["frames"][frame]}
            states = {point[0]: ("estimated" if point[3] else "solid") for point in points}
        boxes = [box for box in boxes if not box[-1]]
        links = match_boxes(
            [np.asarray(box) * np.tile(image_scale, 2) for box, _ in refs],
            [box[1:5] for box in boxes], protocol["contact_protocol"]["box_match_iou"],
        ) if refs and boxes else {}
        extra_solid += len(solid.keys() - {boxes[j][0] for j in links.values()})
        matrix = np.linalg.inv(stage_projection(calibration["frame_corners"][frame]))
        for i, (_, point) in enumerate(refs):
            reference = project_point(matrix, np.asarray(point) * image_scale, stage) if point is not None else None
            estimated = dict.fromkeys(methods)
            raw_diagnostic = None
            if i in links:
                ident, x1, _, x2, y2, _ = boxes[links[i]]
                bottom = project_point(matrix, ((x1 + x2) / 2, y2), stage)
                estimated["box_bottom"] = bottom if np.all((bottom >= 0) & (bottom <= stage)) else None
                estimated["dashboard_solid"] = solid.get(ident)
                if raw_cache:
                    proposal = raw_by_id[ident]
                    if not np.allclose(np.asarray(proposal["box"]) * np.tile(raw_scale, 2),
                                       boxes[links[i]][1:5], atol=.051):
                        raise ValueError(f"Raw and replay boxes differ at frame {frame}, track {ident}")
                    raw_position = project_point(matrix, np.asarray(proposal["point"]) * raw_scale, stage)
                    in_stage = bool(np.all((raw_position >= 0) & (raw_position <= stage)))
                    raw_diagnostic = {
                        "track_id": ident, "reportability": proposal["reportability"],
                        "in_stage": in_stage, "dashboard_state": states.get(ident, "absent"),
                        "above_threshold": proposal["reportability"] >= replay["threshold"],
                        "error_m": float(np.linalg.norm(raw_position - reference)) if reference is not None else None,
                        "raw_position_m": raw_position.tolist(),
                    }
            errors = {
                method: float(np.linalg.norm(value - reference)) if value is not None and reference is not None else None
                for method, value in estimated.items()
            }
            rows.append({
                "frame": frame, "reference_index": i, "has_reference_point": reference is not None,
                "matched_track_box": i in links, "error_m": errors,
                "reported": {method: value is not None for method, value in estimated.items()},
                "stage_positions_m": {key: value.tolist() if value is not None else None
                                      for key, value in {"reference": reference, **estimated}.items()},
                "raw_cache_diagnostic": raw_diagnostic,
            })
    visible = [row for row in rows if row["has_reference_point"]]
    paired = [row for row in visible if all(row["reported"].values())]
    summaries = {}
    for method in methods:
        errors = [row["error_m"][method] for row in visible if row["reported"][method]]
        summaries[method] = {
            "visible_points_reported": len(errors),
            "within_threshold_m": {str(t): sum(error <= t for error in errors)
                                   for t in protocol["contact_protocol"]["map_thresholds_m"]},
            "median_reported_error_m": float(np.median(errors)) if errors else None,
            "paired_median_error_m": float(np.median([row["error_m"][method] for row in paired])) if paired else None,
            "reported_when_reference_uncertain": sum(row["reported"][method] for row in rows if not row["has_reference_point"]),
        }
    result = {
        "dashboard_threshold": replay["threshold"], "reference_points": len(visible),
        "uncertain_reference_points": len(rows) - len(visible), "paired": len(paired),
        "matched_track_boxes": sum(row["matched_track_box"] for row in rows),
        "solid_points_without_matched_observed_box": extra_solid,
        "methods": summaries, "cases": rows,
        "replay_sha256": digest(source.with_suffix(".js")), "calibration_sha256": digest(calibration_file),
    }
    if raw_cache:
        passing = [row for row in visible if (probe := row["raw_cache_diagnostic"]) is not None
                   and probe["in_stage"] and probe["above_threshold"]]
        common = [row for row in passing if all(row["reported"].values())]
        diagnostic = {
            "scope": "Post-hoc comparison of cached raw proposals and published replay points on the same observed track boxes.",
            "cache_sha256": digest(raw_path),
            "visible_with_no_matched_box": sum(not row["matched_track_box"] for row in visible),
            "visible_with_raw_below_threshold": sum(row["raw_cache_diagnostic"] is not None
                                                    and not row["raw_cache_diagnostic"]["above_threshold"] for row in visible),
            "visible_with_raw_in_stage_above_threshold": len(passing),
            "raw_passing_changed_to_estimated_or_absent": sum(not row["reported"]["dashboard_solid"] for row in passing),
            "within_threshold_m": {str(t): sum(row["raw_cache_diagnostic"]["error_m"] <= t for row in passing)
                                   for t in protocol["contact_protocol"]["map_thresholds_m"]},
            "paired_all_three": len(common),
            "paired_medians_m": {
                method: float(np.median([row["raw_cache_diagnostic"]["error_m"] if method == "raw_head" else row["error_m"][method]
                                        for row in common])) if common else None
                for method in (*methods, "raw_head")
            },
            "largest_dashboard_paired_errors": sorted(paired, key=lambda row: row["error_m"]["dashboard_solid"], reverse=True)[:5],
        }
        result["post_hoc_raw_head"] = diagnostic
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--raw-cache", type=Path, help="Optional original Wicked Games inference JSON for the raw-head diagnostic")
    parser.add_argument("--output", type=Path, default=EVIDENCE / "results.json")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    locked = verify_inputs()
    protocol = read_json(EVIDENCE / "protocol.json")
    labels = read_json(EVIDENCE / "annotations.json")
    frames = protocol["frames"]
    annotations = labels["frames"]
    if set(map(int, annotations)) != set(frames):
        raise ValueError("Every predeclared frame must have a reference record")
    images = {frame: cv2.imread(str(EVIDENCE / f"frames/{frame:04d}.png")) for frame in frames}
    settings = protocol["detector_protocol"]
    floor = settings["fixed_confidence"]
    report = {
        "inputs": locked, "device": "CPU" if args.device == "cpu" else torch.cuda.get_device_name(int(args.device)),
        "torch": torch.__version__, "detectors": {}, "ablation": {},
    }
    for name, weight in (("generic", ROOT / "models/yolo26m.pt"), ("published", PUBLISHED), ("tuned_300", FINAL)):
        model = YOLO(weight)
        raw = {frame: candidates(model, image, settings["tile_image_size"], args.device)
               for frame, image in images.items()}
        merged = {frame: merge(rows, settings["merge_iou"]) for frame, rows in raw.items()}
        report["detectors"][name] = {
            "weights_sha256": digest(weight),
            **{split: detection_metrics(annotations, merged, subset, floor)
               for split, subset in {"all": frames, **protocol["strata"]}.items()},
        }
        metric = report["detectors"][name]["all"]
        print(f"{name}: AP50 {metric['ap50']:.3f}, AP50:95 {metric['ap50_95']:.3f}", flush=True)
        if name == "tuned_300":
            alternative = {frame: merge(rows, settings["ablation_merge_iou"]) for frame, rows in raw.items()}
            report["ablation"] = detection_metrics(annotations, alternative, frames, floor)
    report["contacts"] = contact_metrics(protocol, annotations, args.raw_cache)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
