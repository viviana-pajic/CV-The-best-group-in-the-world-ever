"""Project a new inference run to the stage and export a dashboard replay."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from build_dashboard_data import ROOT, read, stage_projection

sys.path.insert(0, str(ROOT / "Experiment/Dance_Overnight/code"))
from trajectory_fit import fit

SELECTION = ROOT / "Experiment/lighting_expansion_300/runs/lighting_300_visual_s17/joint_evaluation/selection.json"
# Keep the published trajectory filter in its original coordinate system.
# The dashboard export then applies the physical 12 × 12 m aspect ratio.
FIT_BOUNDS = np.array([100.0, 62.5])


def save(path, data):
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def project(data, calibration, threshold, settings):
    frames = data["frames"]
    corners = calibration["frame_corners"]
    if len(frames) != data["count"] or len(frames) != len(corners) or data["fps"] != calibration["fps"]:
        raise ValueError("Stage alignment and inference cover different video frames")
    if data["source_resolution"] != calibration["source_resolution"]:
        raise ValueError("Stage alignment and inference use different source resolutions")
    scale = np.asarray(data["source_resolution"]) / data["inference_resolution"]
    ids = sorted({row["id"] for frame in frames for row in frame})
    columns = {ident: i for i, ident in enumerate(ids)}
    raw = np.full((len(frames), len(ids), 2), np.nan)
    confidence = np.zeros(raw.shape[:2])
    tracks = []
    for k, (frame, quad) in enumerate(zip(frames, corners, strict=True)):
        camera = stage_projection(quad)
        matrix = np.linalg.inv(camera) if camera is not None else None
        rows = []
        for row in frame:
            rows.append({"id": row["id"], "box": (np.asarray(row["box"]) * np.tile(scale, 2)).tolist(),
                         "status": row["status"], "score": row["score"]})
            if matrix is None or row["reportability"] < threshold:
                continue
            x, y, w = matrix @ [*(np.asarray(row["point"]) * scale), 1]
            if abs(w) <= 1e-9:
                continue
            relative = np.array([x / w, y / w])
            if ((relative >= 0) & (relative <= 1)).all():
                column = columns[row["id"]]
                raw[k, column] = relative * FIT_BOUNDS
                confidence[k, column] = row["reportability"]
        tracks.append(rows)

    fitted, states, ages = fit(raw, confidence, settings)
    map_frames = []
    for k, rows in enumerate(tracks):
        points = []
        if corners[k] is not None:
            visible_ids = {row["id"] for row in rows}
            for column in np.flatnonzero(states[k]):
                ident = ids[column]
                status = "proxy" if states[k, column] == 1 else "estimated"
                points.append([ident, *fitted[k, column].tolist(), status, int(ages[k, column])])
                if ident not in visible_ids:
                    rows.append({"id": ident, "box": None, "status": "unknown"})
        map_frames.append(points)
    return tracks, map_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    source = args.source.resolve()
    data = read(source.with_suffix(".inference.json"))
    with source.open("rb") as video:
        if file_digest(video, "sha256").hexdigest() != data["config"]["source_sha256"]:
            raise ValueError("Inference came from a different video file")
    calibration_path = source.with_suffix(".calibration.json")
    calibration = read(calibration_path)
    selection = read(SELECTION)
    threshold = selection["display_threshold"]
    tracks, map_frames = project(data, calibration, threshold, selection["settings"])
    track_path = source.with_suffix(".tracks.json")
    map_path = source.with_suffix(".map.json")
    save(track_path, {"frames": tracks})
    save(map_path, {"first_source_frame": 0, "fps": data["fps"], "threshold": threshold,
                    "bounds": {"x": [0, 100], "y": [0, 62.5]},
                    "fields": ["track_id", "x", "y", "map_status", "map_age_frames"], "frames": map_frames})
    counts = {
        "frames": data["count"], "calibrated_frames": calibration["calibrated_frames"],
        "frames_with_boxes": sum(any(row["status"] == "observed" for row in frame) for frame in tracks),
        "frames_with_contacts": sum(bool(frame) for frame in map_frames),
        "reported_contacts": sum(point[3] == "proxy" for frame in map_frames for point in frame),
        "estimated_contacts": sum(point[3] == "estimated" for frame in map_frames for point in frame),
        "gpu": data["device"],
    }
    save(source.with_suffix(".coverage.json"), counts)
    print(f"{args.title}: {counts}", flush=True)
    subprocess.run([
        sys.executable, str(ROOT / "tools/build_dashboard_data.py"),
        "--source", str(source), "--positions", str(map_path), "--tracks", str(track_path),
        "--calibration", str(calibration_path), "--id", source.stem, "--title", args.title,
        "--stage-meters", "12", "12", "--start-frame", str(args.start_frame), "--copy-video",
    ], check=True)


if __name__ == "__main__":
    main()
