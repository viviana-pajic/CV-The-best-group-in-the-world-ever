"""Select extra training frames and propose annotations from nearby visual labels."""
from bisect import bisect_left
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "Experiment/lighting_expansion_200"
sys.path.insert(0, str(PREVIOUS))
from build_dataset import EXP, FIRST, LAST, HOLDOUT, read, video_frames, write

POSE_WEIGHTS = ROOT.parent / "Dance YOLO trial/models/yolo26m-pose.pt"
WIDTH, HEIGHT = 1280, 720


def anchors_for(frame, existing):
    index = bisect_left(existing, frame)
    return [k for k in existing[max(0, index - 1):index + 1] if abs(frame - k) <= 7]


def frame_features(image, tracks):
    small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    return [np.median(gray[18:44, 10:150]), np.median(gray[48:68, 15:145]),
            np.median(sat[20:62, 10:150]), sum(row["status"] == "observed" for row in tracks)]


def choose(existing, validation, tracks):
    features = {}
    for frame, image in video_frames():
        if frame in existing or (
            frame not in HOLDOUT and all(abs(frame - k) > 10 for k in validation)
            and anchors_for(frame, existing) and min(abs(frame - k) for k in anchors_for(frame, existing)) >= 2
        ):
            features[frame] = frame_features(image, tracks[frame - FIRST])
    frames = np.array(sorted(features))
    raw = np.array([features[k] for k in frames])
    scaled = (raw - np.median(raw, axis=0)) / np.maximum(np.ptp(np.percentile(raw, [10, 90], axis=0), axis=0), 1)
    space = np.column_stack(((frames - FIRST) / 120, np.clip(scaled, -2, 2)))
    existing_mask = np.isin(frames, existing)
    distance = np.min(np.sum((space[:, None] - space[existing_mask]) ** 2, axis=2), axis=1)
    proximity = np.array([0 if k in existing else min(abs(k - a) for a in anchors_for(k, existing)) for k in frames])
    eligible = ~existing_mask & (raw[:, 3] > 0)
    selected = []
    segments = np.array_split(np.arange(FIRST, LAST + 1), 12)
    while len(selected) < 100:
        before = len(selected)
        for segment in segments:
            valid = np.flatnonzero(eligible & (frames >= segment[0]) & (frames <= segment[-1]))
            if valid.size == 0:
                continue
            # Reward novel lighting and crowd conditions, with a penalty for distant anchors.
            score = distance[valid] / np.maximum(proximity[valid], 2) ** 0.7
            index = valid[np.argmax(score)]
            selected.append(int(frames[index]))
            distance = np.minimum(distance, np.sum((space - space[index]) ** 2, axis=1))
            eligible &= np.abs(frames - frames[index]) >= 6
            if len(selected) == 100:
                break
        if before == len(selected):
            raise RuntimeError("Not enough separated training frames")
    return sorted(selected), {str(k): [float(v) for v in features[k]] for k in selected}


def flow(source, target, pts):
    pts = np.float32(pts).reshape(-1, 1, 2)
    options = {"winSize": (19, 19), "maxLevel": 3,
               "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}
    forward, found, _ = cv2.calcOpticalFlowPyrLK(source, target, pts, None, **options)
    back, found_back, _ = cv2.calcOpticalFlowPyrLK(target, source, forward, None, **options)
    forward = forward[:, 0]
    valid = (found[:, 0] & found_back[:, 0]).astype(bool)
    valid &= np.linalg.norm(back[:, 0] - pts[:, 0], axis=1) < 1.8
    valid &= ((forward >= 0) & (forward < [WIDTH, HEIGHT])).all(axis=1)
    return forward, valid


def warp(source, target, rows, anchor):
    points, locations = [], []
    for row in rows:
        x1, y1, x2, y2 = row["box"]
        xs = np.linspace(x1 + .2 * (x2 - x1), x2 - .2 * (x2 - x1), 4)
        ys = np.linspace(y1 + .12 * (y2 - y1), y2 - .10 * (y2 - y1), 5)
        start = len(points)
        points.extend((x, y) for x in xs for y in ys)
        if row["floor"] is not None:
            points.append(row["floor"])
        locations.append((start, len(points)))
    moved, valid = flow(source, target, points)
    result = []
    for row, (start, stop) in zip(rows, locations, strict=True):
        mask = valid[start:start + 20]
        if mask.sum() < 7:
            continue
        delta = moved[start:start + 20][mask] - np.asarray(points[start:start + 20])[mask]
        shift = np.median(delta, axis=0)
        if np.median(np.linalg.norm(delta - shift, axis=1)) > 7:
            continue
        box = np.asarray(row["box"]) + np.tile(shift, 2)
        point = None
        if row["floor"] is not None and valid[stop - 1]:
            candidate = moved[stop - 1]
            if np.linalg.norm(candidate - (np.array(row["floor"]) + shift)) <= max(8, .18 * (box[3] - box[1])):
                point = candidate
        result.append({"box": box, "point": point, "contact": row["contact"],
                       "kind": row["support_type"], "anchor": anchor})
    return result


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - overlap
    return overlap / max(1, union)


def links(targets, others, threshold):
    if not targets or not others:
        return {}
    overlap = np.array([[iou(a["box"], b["box"]) for b in others] for a in targets])
    ii, jj = linear_sum_assignment(1 - overlap)
    return {int(i): int(j) for i, j in zip(ii, jj) if overlap[i, j] >= threshold}


def distance(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))


def ankle_point(pose, box):
    if pose is None:
        return None
    keypoints = pose["keypoints"]
    feet = []
    for hip, knee, ankle in [(11, 13, 15), (12, 14, 16)]:
        h, k, a = keypoints[[hip, knee, ankle]]
        if min(h[2], k[2], a[2]) < .55:
            continue
        first, second = k[:2] - h[:2], a[:2] - k[:2]
        scale = np.linalg.norm(first) * np.linalg.norm(second)
        if scale > 1 and (first @ second) / scale > .5 and a[1] > k[1] + 3:
            feet.append(a[:2] + [0, min(3, .035 * (box[3] - box[1]))])
    if not feet:
        return None
    if len(feet) == 2 and abs(feet[0][1] - feet[1][1]) <= 6:
        return np.mean(feet, axis=0)
    return max(feet, key=lambda point: point[1])


def contact_for(nearby, models, ankle, box):
    height = box[3] - box[1]
    tolerance = max(7, min(13, .11 * height))
    visible = [row for row in nearby if row["point"] is not None]
    best = visible[0] if visible else None
    reliable_models = [p["point"] for p in models if p.get("point") is not None
                       and p["reportability"] >= .72 and p["augmentation_rms_pixels"] <= 6]
    if best:
        point = best["point"]
        kind = best["kind"]
        same = [row["point"] for row in visible[1:]
                if row["kind"] == kind and distance(row["point"], point) <= tolerance]
        if same:
            return "visible", np.mean([point, *same], axis=0), kind, "two_flow_anchors"
        if any(distance(p, point) <= tolerance for p in reliable_models):
            return "visible", point, kind, "flow_and_support_model"
        if kind == "feet" and ankle is not None and distance(ankle, point) <= tolerance:
            return "visible", point, kind, "flow_and_pose"
    if ankle is not None and any(distance(p, ankle) <= tolerance for p in reliable_models):
        return "visible", ankle, "feet", "pose_and_support_model"
    if len(nearby) >= 2 and all(row["contact"] == nearby[0]["contact"] for row in nearby):
        if nearby[0]["contact"] in {"hidden", "airborne"}:
            return nearby[0]["contact"], None, None, "two_anchor_status"
    if nearby and nearby[0]["contact"] == "hidden" and not reliable_models and ankle is None:
        return "hidden", None, None, "nearby_hidden_status"
    return "uncertain", None, None, "needs_source_review"


def build_rows(tracks, warped, poses, models, frame):
    observed = [row for row in tracks if row["status"] == "observed" and row["box"] is not None]
    anchors = [links(observed, rows, .32) for rows in warped]
    pose_links = links(observed, poses, .30)
    rows = []
    for i, row in enumerate(observed):
        ident = row["id"]
        near = sorted([a[m[i]] for a, m in zip(warped, anchors) if i in m],
                      key=lambda item: abs(item["anchor"] - frame))
        pose = poses[pose_links[i]] if i in pose_links else None
        box = np.asarray(row["box"]).copy()
        if near and iou(box, near[0]["box"]) > .58:
            box = .65 * box + .35 * near[0]["box"]
        model = [p.get(str(ident), {}) for p in models]
        status, point, kind, method = contact_for(near, model, ankle_point(pose, box), box)
        if point is not None:
            gap = np.maximum(0, np.maximum(box[:2] - point, point - box[2:]))
            if max(gap) > 15 or np.any(point < 0) or np.any(point >= [WIDTH, HEIGHT]):
                status, point, kind, method = "uncertain", None, None, "outlying_point"
            else:
                box[:2] = np.minimum(box[:2], point - 1)
                box[2:] = np.maximum(box[2:], point + 1)
        box = np.clip(box, 0, [WIDTH, HEIGHT, WIDTH, HEIGHT])
        if min(box[2:] - box[:2]) < 4:
            continue
        rows.append({"id": ident, "box": box.round(2).tolist(), "floor": point.round(2).tolist() if point is not None else None,
                     "contact": status, "support_type": kind, "sample_split": "train",
                     "outside": False, "reviewed": False, "provenance": f"Assisted proposal: {method}."})
    return rows


def pose_by_frame(selected):
    from ultralytics import YOLO

    model = YOLO(str(POSE_WEIGHTS))
    paths = [str(OUT / "frames" / f"frame_{k:06}.png") for k in selected]
    results = model.predict(paths, stream=True, device=0, imgsz=960, batch=8, conf=.2, verbose=False)
    return {frame: [{"box": box, "keypoints": points} for box, points in zip(
        result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy(), strict=True)]
        for frame, result in zip(selected, results, strict=True)}


def sheet(image, rows, frame, path):
    # Crop to the stage region without scaling coordinates.
    colors = {"visible": (195, 236, 85), "uncertain": (49, 178, 255),
              "hidden": (186, 151, 239), "airborne": (100, 122, 255)}
    margin = 38
    top = min(265, max(0, int(min(row["box"][1] for row in rows)) - 12))
    bottom = min(HEIGHT, max(616, int(max(row["box"][3] for row in rows)) + 12))
    raw = image[top:bottom]
    marked = image.copy()
    for row in rows:
        x1, y1, x2, y2 = np.rint(row["box"]).astype(int)
        color = colors[row["contact"]]
        cv2.rectangle(marked, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        title = str(row["id"])
        cv2.rectangle(marked, (x1, y1 - 13), (x1 + 8 * len(title) + 4, y1), (30, 25, 22), -1)
        cv2.putText(marked, title, (x1 + 1, y1 - 2), 0, .40, color, 1, cv2.LINE_AA)
        if row["floor"] is not None:
            xy = tuple(np.rint(row["floor"]).astype(int))
            cv2.circle(marked, xy, 4, (22, 20, 19), -1, cv2.LINE_AA)
            cv2.circle(marked, xy, 2, color, -1, cv2.LINE_AA)
    canvas = np.full((len(raw) * 2 + margin * 2, WIDTH, 3), (26, 22, 20), np.uint8)
    canvas[margin:margin + len(raw)] = raw
    canvas[2 * margin + len(raw):] = marked[top:bottom]
    cv2.putText(canvas, f"{frame}  SOURCE   |   crop y={top}..{bottom}   |   x=0..1279", (13, 25), 0, .60,
                (243, 240, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "PROPOSALS   cyan: visible   orange: uncertain   purple: hidden   salmon: airborne",
                (13, margin + len(raw) + 25), 0, .56, (243, 240, 235), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    cv2.setNumThreads(4)
    old = read(PREVIOUS / "visual_relabel/visual_labels.json")
    existing = sorted(map(int, old))
    dev = read(EXP / "data/development_ending.json")
    validation = sorted(d["source_first"] + k for d in dev.values() for k in d["val"])
    tracks = read(EXP / "late_full_replay/base_tracks.json")["frames"]
    old_model = read(EXP / "late_full_replay/support/predictions.json")
    new_model = read(PREVIOUS / "runs/lighting_200_visual_s17/evaluation/predictions.json")
    selected, features = choose(existing, validation, tracks)
    # Interleave in source order so each review batch spans the choreography.
    assignment = {key: sorted(existing + selected)[index::3] for index, key in enumerate("ABC")}
    write(OUT / "assignments.json", {"assignments": assignment, "existing_frames": existing,
          "new_frames": selected, "validation_frames": validation, "features_by_new_frame": features,
          "selection": "Lighting, crowd and time coverage. New frames are separated by at least six frames and 2–7 frames from a previously revised anchor. Training-only selection."})
    images_dir = OUT / "frames"
    images_dir.mkdir(exist_ok=True)
    for frame in existing:
        name = f"frame_{frame:06}.png"
        target = images_dir / name
        if not target.exists():
            target.hardlink_to(PREVIOUS / "frames" / name)
    anchor_map = {k: anchors_for(k, existing) for k in selected}
    want = set(selected).union(a for seq in anchor_map.values() for a in seq)
    gray = {}
    for k, image in video_frames():
        if k in want:
            gray[k] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if k in anchor_map:
            assert cv2.imwrite(str(images_dir / f"frame_{k:06}.png"), image)
    print("Selected and extracted", len(selected), "new frames", flush=True)
    poses = pose_by_frame(selected)
    sheets = OUT / "qa_source"
    sheets.mkdir(exist_ok=True)
    additions = {}
    for frame in selected:
        warped = [warp(gray[a], gray[frame], old[str(a)], a) for a in anchor_map[frame]]
        k = frame - FIRST
        rows = build_rows(tracks[k], warped, poses[frame], [old_model[k], new_model[k]], frame)
        assert rows
        additions[str(frame)] = rows
        path = images_dir / f"frame_{frame:06}.png"
        sheet(cv2.imread(str(path)), rows, frame, sheets / f"{frame:06}.jpg")
    write(OUT / "new_proposals.json", additions)
    print("Draft", Counter(row["contact"] for rows in additions.values() for row in rows), flush=True)


if __name__ == "__main__":
    main()
