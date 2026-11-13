"""Compare one visible onare row against visual references at the same frame."""
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/final_formation_validation"
MATCH_RADIUS_PX = 30


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def node(code, data):
    return json.loads(subprocess.run(["node", "-e", code], input=json.dumps(data), text=True,
                                     capture_output=True, check=True, cwd=ROOT).stdout)


def main():
    reference = read(EVIDENCE / "onare_reference.json")
    ident, frame = reference["id"], reference["frame"]
    original = node("""
        const fs = require('node:fs');
        const vm = require('node:vm');
        require('./dashboard/formation.js');
        const {id, frame} = JSON.parse(fs.readFileSync(0, 'utf8'));
        const context = {window: {}};
        vm.runInNewContext(fs.readFileSync('dashboard/assets/' + id + '.js', 'utf8'), context);
        const clip = context.window.DANCE_CLIPS[0];
        const positions = clip.frames.map(record => record[1]);
        const lines = createFormationEngine().analyze(positions, clip.fps, 4, 10)[frame];
        console.log(JSON.stringify({resolution: clip.resolution, positions: positions[frame], lines}));
        """, {"id": ident, "frame": frame})
    ref_px = np.float32(reference["points_px"])
    scale = np.asarray(reference["resolution"]) / original["resolution"]
    contacts = np.asarray(original["positions"])
    detected_px = contacts[:, 4:6] * scale
    distance = np.linalg.norm(ref_px[:, None, :] - detected_px[None, :, :], axis=2)
    left, right = linear_sum_assignment(distance)
    matched = [(i, j) for i, j in zip(left, right, strict=True) if distance[i, j] <= MATCH_RADIUS_PX]
    by_id = {int(contacts[j, 0]): i for i, j in matched}

    calibration = read(ROOT / "experiments/replays" / f"{ident}.calibration.json")
    corners = np.float32(np.asarray(calibration["frame_corners"][frame]) * scale)
    matrix = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [100, 0], [100, 100], [0, 100]]))
    stage = cv2.perspectiveTransform(ref_px[None], matrix)[0]
    line, = original["lines"]
    line_ids = [point[0] for point in line["members"]]
    same_reference = [by_id[person] for person in line_ids if person in by_id]
    reference_rows = [[i + 1, *map(float, stage[i]), 0] for i in range(len(stage))]
    visual_scores = node("""
        const fs = require('node:fs');
        require('./dashboard/formation.js');
        const {all, subset} = JSON.parse(fs.readFileSync(0, 'utf8'));
        const engine = createFormationEngine();
        const measure = points => engine.analyze(Array.from({length: 32}, () => points), 25, 4, 10)[12]
            .map(row => ({angle: row.angle, score: row.score, members: row.members.length}));
        console.log(JSON.stringify({all: measure(all), subset: measure(subset)}));
        """, {"all": reference_rows, "subset": [reference_rows[i] for i in same_reference]})
    matched_reported = [(i, j) for i, j in matched if contacts[j, 3] == 0]
    result = {
        "clip": reference["clip"],
        "frame": frame,
        "time_s": frame / 25,
        "reference_dancers": len(ref_px),
        "reported_contacts_within_30_px": len(matched_reported),
        "estimated_positions_within_30_px": len(matched) - len(matched_reported),
        "mean_reported_contact_error_m": round(float(np.mean(
            [np.linalg.norm(contacts[j, 1:3] - stage[i]) * .12 for i, j in matched_reported])), 3),
        "dashboard_line_dancers": len(line_ids),
        "dashboard_line_dancers_with_reference": len(same_reference),
        "dashboard_line_angle_deg": round(line["angle"], 2),
        "dashboard_line_score_pct": round(line["score"], 2),
        "same_dancers_visual_reference": visual_scores["subset"],
        "all_visual_references": visual_scores["all"],
        "scope": "One illustrative frame. Reference-only inputs are repeated to isolate spatial fitting. The dashboard result uses the full replay with temporal context.",
    }
    (EVIDENCE / "onare_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
