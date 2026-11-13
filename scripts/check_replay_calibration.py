"""Compare static and registered calibration on visible mat edges."""
from pathlib import Path
import json
import math
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/final_formation_validation"
ORIGINAL = ROOT / "Experiment/Dance_CVAT_Pilot/calibration_draft.json"
REPLAYS = ROOT / "experiments/replays"
BOUNDARIES = {"back": (0, 1), "right": (1, 2), "front": (2, 3), "left": (3, 0)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    return abs(dx * (a[1] - point[1]) - dy * (a[0] - point[0])) / math.hypot(dx, dy)


def edge_errors(edges, corners):
    return {
        name: [distance(point, corners[a], corners[b]) for point in points]
        for name, points in edges.items()
        for a, b in [BOUNDARIES[name]]
    }


def main():
    picks = read(EVIDENCE / "edge_picks.json")
    original = read(ORIGINAL)["image_points"]
    table = []
    for case in picks["cases"]:
        registration = read(REPLAYS / f"{case['id']}.calibration.json")
        width, height = registration["source_resolution"]
        registered = [[x * 1280 / width, y * 720 / height]
                      for x, y in registration["frame_corners"][case["frame"]]]
        old = edge_errors(case["edges"], original)
        new = edge_errors(case["edges"], registered)
        table.append({
            "clip": case["clip"],
            "frame": case["frame"],
            "points": sum(map(len, case["edges"].values())),
            "static_median_px": round(median(error for values in old.values() for error in values), 2),
            "registered_median_px": round(median(error for values in new.values() for error in values), 2),
            "edge_medians_px": {name: {"static": round(median(old[name]), 2), "registered": round(median(new[name]), 2)}
                                for name in old},
        })
    result = {"method": picks["method"], "cases": table}
    (EVIDENCE / "registration_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
