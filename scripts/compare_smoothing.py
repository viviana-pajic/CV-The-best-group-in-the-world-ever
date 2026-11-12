"""Compare raw contacts, causal Kalman, and offline robust fit on frozen inputs."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EARLY = ROOT / "Experiment/Dance_Overnight"
sys.path.insert(0, str(EARLY / "code"))
from evaluate_final_confirmation import score
from filter_lab import continuity, export, inputs
import frozen_holdout as fh
from trajectory_fit import fit


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def kalman(raw, gap):
    output = np.full_like(raw, np.nan)
    states = np.zeros(raw.shape[:2], dtype=int)
    ages = np.full(raw.shape[:2], 999, dtype=int)
    dt = 1 / 25
    G = np.float32([[dt**2/2, 0], [0, dt**2/2], [dt, 0], [0, dt]])
    for j in range(raw.shape[1]):
        track = cv2.KalmanFilter(4, 2)
        track.transitionMatrix = np.float32([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        track.measurementMatrix = np.eye(4, dtype=np.float32)[:2]
        track.processNoiseCov = G @ G.T * 30**2 + np.eye(4, dtype=np.float32) * 1e-6
        track.measurementNoiseCov = np.eye(2, dtype=np.float32) * .6**2
        last = -999
        for k, point in enumerate(raw[:, j]):
            known = np.isfinite(point).all()
            if k - last > gap:
                if not known:
                    continue
                track.statePost = np.float32([*point, 0, 0])[:, None]
                track.errorCovPost = np.diag(np.float32([.36, .36, 100., 100.]))
                value = point
            else:
                value = track.predict().ravel()[:2]
                if known:
                    value = track.correct(point.astype(np.float32).reshape(2, 1)).ravel()[:2]
            if known:
                last = k
            if 0 <= value[0] <= 100 and 0 <= value[1] <= 62.5:
                output[k, j] = value
                states[k, j] = 1 if known else 2
                ages[k, j] = k-last
    return output, states, ages


def main():
    cv2.setNumThreads(2)
    selection = load(EARLY / "final_confirmation/frozen_selection.json")
    ref = load(EARLY / "late_full_replay/provenance.json")
    base = load(EARLY / ref["base_tracks"])
    predictions = load(EARLY / ref["support_predictions"])
    fh.D = EARLY / "data/final_confirmation"
    labels = load(fh.D / "labels.json")
    raw, confidence, ids = inputs(base, predictions, selection["threshold"])
    known = np.isfinite(raw[:,:,0])
    methods = {"Raw": (raw, known.astype(int), np.where(known, 0, 999)),
               "Causal Kalman": kalman(raw, 25),
               "Offline robust": fit(raw, confidence, selection["settings"])}
    shared = np.logical_and.reduce([np.isfinite(value[0][:,:,0]) for value in methods.values()])
    result = {}
    for name, (points, state, age) in methods.items():
        scores = score(export(base, points, state, age, raw, ids, selection["settings"]), labels)["all"]
        common = continuity(points, raw, shared)["candidate"]
        result[name] = {"within_3pct": scores["correct_by_tolerance_pct_width"]["3"],
                        "visible_refs": scores["visible_references"], "hidden_measured": scores["hidden_measured"],
                        "continuity_on_common_samples": common}
    step = np.full((90, 1, 2), [35., 25.])
    step[45:, 0, 0] += 3
    traces = {"Raw": step, "Causal Kalman": kalman(step, 25)[0],
              "Offline robust": fit(step, np.ones((90,1)), {**selection["settings"], "path_gate": False})[0]}
    for name, trace in traces.items():
        result[name]["synthetic_3pct_step_half_response_frames"] = int(np.flatnonzero(trace[:,0,0] >= 36.5)[0])-45
    output = ROOT / "experiments/tracking/smoothing.json"
    output.write_text(json.dumps({"results": result, "scope": "Frozen boxes and support observations; diagnostic reserved references. The synthetic step isolates the smoother and disables the offline path gate."}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
