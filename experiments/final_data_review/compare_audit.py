"""Compare frozen blind audit observations with the merged training annotations."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MERGED = ROOT / "Experiment/lighting_expansion_300/merged_annotations.xml"
BLIND = HERE / "blind_observations.csv"
SAMPLE = HERE / "sample.csv"
BOX_FIELDS = ("x1", "y1", "x2", "y2")
MERGED_BOX_FIELDS = ("xtl", "ytl", "xbr", "ybr")
MATCH_IOU = 0.5
AMBIGUITY_MARGIN = 0.10


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {int(row["source_frame"]): row for row in csv.DictReader(f)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merged_rows():
    result = {}
    for frame in ET.parse(MERGED).findall("image"):
        frame_number = int(Path(frame.get("name")).stem.rsplit("_", 1)[-1])
        points = {shape.findtext("attribute[@name='instance_id']"): tuple(map(float, shape.get("points").split(",")))
                  for shape in frame.findall("points[@label='floor_support']")}
        result[frame_number] = [
            {"id": shape.findtext("attribute[@name='instance_id']"),
             "box": tuple(float(shape.get(key)) for key in MERGED_BOX_FIELDS),
             "contact": shape.findtext("attribute[@name='contact']"),
             "support_type": shape.findtext("attribute[@name='support_type']"),
             "point": points.get(shape.findtext("attribute[@name='instance_id']"))}
            for shape in frame.findall("box[@label='dancer']")
        ]
    return result


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return intersection / (area(a) + area(b) - intersection)


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (index - low) * (values[high] - values[low])


def compare(sample, blind, prior):
    comparisons = []
    for frame, review in blind.items():
        contact = review["contact_state"]
        record = {"source_frame": frame, "batch": sample[frame]["batch"], "blind_contact": contact,
                  "blind_box": None, "blind_point": None, "blind_support_type": review["support_type"] or None,
                  "blind_note": review["observation"], "match": "abstained"}
        if contact == "indistinguishable":
            assert not any(review[key] for key in (*BOX_FIELDS, "point_x", "point_y"))
            comparisons.append(record)
            continue
        box = tuple(float(review[key]) for key in BOX_FIELDS)
        assert 0 <= box[0] < box[2] <= 1280 and 0 <= box[1] < box[3] <= 720
        point = tuple(float(review[key]) for key in ("point_x", "point_y")) if contact == "visible" else None
        assert point is None or (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])
        ranked = sorted(prior[frame], key=lambda row: iou(box, row["box"]), reverse=True)
        best, runner_up = ranked[:2]
        best_iou, runner_up_iou = (iou(box, row["box"]) for row in (best, runner_up))
        status = "below_threshold" if best_iou < MATCH_IOU else (
            "ambiguous" if best_iou - runner_up_iou < AMBIGUITY_MARGIN else "matched")
        prior_point = best["point"]
        review_support = "mixed_knee_foot" if review["support_type"] == "mixed knee and foot" else review["support_type"]
        record.update(blind_box=box, blind_point=point, match=status, candidate_id=best["id"],
                      prior_box=best["box"], prior_point=prior_point, prior_contact=best["contact"],
                      prior_support_type=best["support_type"], box_iou=best_iou, runner_up_iou=runner_up_iou,
                      runner_up_id=runner_up["id"], contact_agrees=contact == best["contact"],
                      support_agrees=review_support == best["support_type"] if point and prior_point else None,
                      point_distance_px=math.dist(point, prior_point) if point and prior_point else None)
        comparisons.append(record)
    return comparisons


def summarize(comparisons):
    matched = [row for row in comparisons if row["match"] == "matched"]
    points = [row["point_distance_px"] for row in matched if row["point_distance_px"] is not None]
    ious = [row["box_iou"] for row in matched]
    return {"frames_reviewed": len(comparisons), "blind_box_count": sum(row["blind_box"] is not None for row in comparisons),
            "matching_counts": dict(Counter(row["match"] for row in comparisons)),
            "box_iou_median": statistics.median(ious) if ious else None,
            "box_iou_p10": quantile(ious, .10),
            "contact_state_agreement": sum(row["contact_agrees"] for row in matched),
            "contact_state_denominator": len(matched),
            "joint_visible_points": len(points),
            "support_type_agreement": sum(row["support_agrees"] is True for row in matched),
            "point_distance_median_px": statistics.median(points) if points else None,
            "point_distance_p90_px": quantile(points, .90),
            "points_within_10_px": sum(value <= 10 for value in points)}


def main():
    sample, blind = rows(SAMPLE), rows(BLIND)
    assert sample.keys() == blind.keys() and len(sample) == 36
    comparisons = compare(sample, blind, merged_rows())
    payload = {
        "scope": "One selected target per training frame, one AI visual pass against the existing merged export.",
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in (SAMPLE, BLIND, MERGED, HERE / "targets.csv")},
        "matching": {"minimum_iou": MATCH_IOU, "minimum_iou_margin_over_runner_up": AMBIGUITY_MARGIN},
        "overall": summarize(comparisons),
        "by_batch": {batch: summarize([row for row in comparisons if row["batch"] == batch]) for batch in "ABC"},
        "per_frame": comparisons,
    }
    (HERE / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
