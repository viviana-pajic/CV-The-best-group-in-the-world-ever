"""Build review figures and data audits from saved results and source frames."""
from csv import DictWriter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Experiment"
TRIAL = EXP / "lighting_expansion_300"
EARLY = EXP / "Dance_Overnight"
OUT = ROOT / "experiments"
VIDEO = ROOT / "media/source_choreography.mp4"
RUN = TRIAL / "runs/lighting_300_visual_s17/joint_evaluation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def load_frames(keys):
    cap = cv2.VideoCapture(str(VIDEO))
    result = {}
    try:
        for key in sorted(keys):
            cap.set(cv2.CAP_PROP_POS_FRAMES, key)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Could not decode {key}")
            result[key] = image
    finally:
        cap.release()
    return result


def data_audit():
    manifest = read(TRIAL / "training_manifest.json")["lighting_300"]
    train, val = set(manifest["train"]), set(manifest["val"])
    assignments = read(TRIAL / "assignments.json")["assignments"]
    refs = {int(i) for i in read(EARLY / "data/final_confirmation/labels.json")}
    assert len(train) == 300 and len(val) == 11 and len(refs) == 7
    assert not train & (val | refs | set(range(3938, 4093)))
    assert not val & refs and min(abs(a-b) for a in train for b in val) > 10
    assert set().union(*(set(rows) for rows in assignments.values())) == train
    assert all(len(rows) == 100 for rows in assignments.values())
    audit = read(TRIAL / "structural_qa.json")
    recorded_hashes = audit["training_image_sha256"]
    frames = load_frames(val | refs)
    dev = {k: frames[k] for k in sorted(val)}
    reserved = {k: frames[k] for k in sorted(refs)}
    all_images = [*sorted((EXP / "lighting_expansion_200/frames").glob("*.png")),
                  *sorted((TRIAL / "new_frames").glob("*.png"))]
    by_frame = {int(path.stem.split("_")[-1]): path for path in all_images}
    assert train <= by_frame.keys()
    def signature(image):
        patch = cv2.cvtColor(image[290:605, 120:1160], cv2.COLOR_BGR2GRAY)
        return cv2.resize(patch, (64, 24), interpolation=cv2.INTER_AREA).flatten().astype(float)
    train_signatures, hashes, image_rows = [], set(), []
    by_batch = {k: batch for batch, keys in assignments.items() for k in keys}
    for k in sorted(train):
        path = by_frame[k]
        digest = sha256(path.read_bytes()).hexdigest()
        assert digest == recorded_hashes[str(k)]
        hashes.add(digest)
        image = cv2.imread(str(path))
        patch = signature(image)
        train_signatures.append(patch)
        rows = manifest["labels"][str(k)]
        visible = [r for r in rows if r["floor"] is not None]
        image_rows.append({"source_frame": k, "seconds": k / 25, "batch": by_batch[k],
                           "revision": "200-frame revision" if "lighting_expansion_200" in path.parts else "100-frame expansion",
                           "mean_stage_luma": round(float(patch.mean()), 1),
                           "stage_laplacian_variance": round(float(cv2.Laplacian(
                               cv2.cvtColor(image[290:605, 120:1160], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 1),
                           "boxes": sum(r["box"] is not None for r in rows), "visible": len(visible),
                           "hidden": sum(r["contact"] == "hidden" for r in rows),
                           "airborne": sum(r["contact"] == "airborne" for r in rows),
                           "uncertain": sum(r["contact"] == "uncertain" for r in rows),
                           "visible_support_types": "|".join(sorted({r["support_type"] for r in visible if r["support_type"]})),
                           "sha256": digest, "path": path.relative_to(ROOT).as_posix()})
    with (ROOT / "data/frame_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = DictWriter(stream, fieldnames=image_rows[0].keys())
        writer.writeheader()
        writer.writerows(image_rows)
    def proximity(sample):
        result = []
        for key, image in sample.items():
            raw = signature(image)
            differences = np.abs(np.asarray(train_signatures) - raw).mean(axis=1)
            index = int(differences.argmin())
            result.append({"source_frame": key, "nearest_training_frame": sorted(train)[index],
                           "mean_absolute_luma_difference": round(float(differences[index]), 3)})
        return result
    write(OUT / "data/split_audit.json", {"training": len(train), "validation": len(val), "reserved": len(refs),
          "reserved_excluded_inclusive": [3938, 4092], "minimum_train_validation_gap_frames": min(abs(a-b) for a in train for b in val),
          "unique_verified_training_files": len(hashes),
          "nearest_luma_signature": {"development": proximity(dev), "reserved": proximity(reserved)},
          "scope": "Stage crop resized to 64 × 24 in grayscale. Pixel difference is a screen for similar appearances; same-video correlation remains."})


def geometry():
    calib = read(EXP / "Dance_CVAT_Pilot/calibration_draft.json")
    matrix = np.asarray(calib["pixel_to_relative_mat_homography"])
    pixels = np.float32([r["point_px"] for r in calib["check_observations"]])
    mapped = cv2.perspectiveTransform(pixels[None], matrix)[0]
    distances = []
    for (x, y), item in zip(mapped, calib["check_observations"], strict=True):
        index, boundary = {"left": (0, 0), "right": (0, 1), "back": (1, 0), "front": (1, 1)}[item["edge"]]
        distances.append(abs([float(x), float(y)][index] - boundary))
    write(OUT / "geometry/edge_check.json", {"independent_edge_picks_same_image": len(distances),
          "baseline_median_pixel_edge_error": calib["baseline_median_edge_distance_px"],
          "draft_median_pixel_edge_error": calib["draft_median_edge_distance_px"],
          "median_normalized_edge_distance_pct_width": median(distances) * 100,
          "inverse_identity_max_error": float(np.abs(np.linalg.inv(matrix) @ matrix - np.eye(3)).max()),
          "scope": "Distance to the named edge using held-out pixel picks; front corners were extrapolated. No surveyed landmark error."})


def occupancy():
    frames = read(RUN / "map_positions.json")["frames"]
    maps = {}
    statistics = {}
    for status, label in [("proxy", "Reported contacts"), ("estimated", "Estimated positions")]:
        grid = np.zeros((48, 48))
        covered, spacings = 0, []
        for frame in frames:
            points = np.asarray([[x*.12, y*12/62.5] for _, x, y, state, _ in frame if state == status])
            if not len(points):
                continue
            covered += 1
            counts, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=48, range=((0, 12), (0, 12)))
            grid += counts / len(points)
            if len(points) >= 2 and status == "proxy":
                pairwise = cdist(points, points)
                np.fill_diagonal(pairwise, np.inf)
                spacings.append(float(np.median(pairwise.min(axis=1))))
        if not covered:
            raise ValueError(f"No {status} points for this analysis")
        maps[label] = gaussian_filter(grid / covered, sigma=.6 / .25, mode="reflect")
        statistics[status] = {"covered_frames": covered, "all_frames": len(frames),
                              "median_frame_nearest_neighbor_m": median(spacings) if spacings else None}
    path = OUT / "analytics"
    path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.55), layout="constrained")
    vmax = max(float(grid.max()) for grid in maps.values())
    for axis, (label, grid) in zip(axes, maps.items(), strict=True):
        plot = axis.imshow(grid * 100, extent=(0, 12, 12, 0), cmap="magma", vmin=0, vmax=vmax*100)
        axis.set(title=label, xlabel="Stage width (m)", ylabel="Depth from back (m)", xticks=range(0, 13, 2), yticks=range(0, 13, 2))
    fig.colorbar(plot, ax=axes, label="Average share per 0.25 m cell (%)")
    fig.suptitle("Where the dancers appear on the stage")
    fig.savefig(path / "stage_occupancy.png", dpi=155)
    plt.close(fig)
    write(path / "summary.json", {"statistics": statistics, "cell_m": .25, "bandwidth_m": .6,
          "scope": "Every covered frame contributes equal mass within its status. Gaussian smoothing reflects at stage edges. Conditional on detected points and excludes frames without that status."})


def detector_cases():
    report = read(OUT / "detector/comparison.json")
    keys = [3953, 3999, 4021, 4043, 4061, 4079]
    source = load_frames(keys)
    output = OUT / "error_analysis"
    output.mkdir(parents=True, exist_ok=True)
    for key in keys:
        panels = []
        for model in ("published", "tuned_300"):
            case = report["models"][model]["reserved_diagnostic"]["cases"][str(key)]
            image = source[key].copy()
            for j, row in enumerate(case["predictions"]):
                match = str(j) in case["matched"]
                a, b, c, d = map(round, row["box"])
                color = (100, 210, 95) if match else (45, 174, 254)
                cv2.rectangle(image, (a,b), (c,d), color, 1 if match else 2)
                if not match:
                    cv2.putText(image, f"P{j}", (a, b-5), 0, .44, color, 1, cv2.LINE_AA)
            for j in case["missed"]:
                a,b,c,d = map(round, case["references"][j]["box"])
                cv2.rectangle(image, (a,b), (c,d), (235, 104, 237), 2)
                cv2.putText(image, f"R{j}", (a, d+13), 0, .44, (235, 104, 237), 1, cv2.LINE_AA)
            panel = image[220:635, 125:1045].copy()
            label = f"{model} | frame {key} | unmatched predictions amber, missed references pink"
            cv2.rectangle(panel, (0, 0), (920, 38), (24, 24, 24), -1)
            cv2.putText(panel, label, (15, 25), 0, .62, (255,255,255), 1, cv2.LINE_AA)
            panels.append(panel)
        cv2.imwrite(str(output / f"frame_{key}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    (ROOT / "data").mkdir(exist_ok=True)
    detector_cases()
    geometry()
    occupancy()
    data_audit()
