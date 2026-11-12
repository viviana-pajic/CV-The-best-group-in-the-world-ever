"""Summarize and package the completed 300-image trial."""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = OUT / "runs/lighting_300_visual_s17"
DETECTOR = OUT / "runs/detector_300_finetune_s17"
OLD_RUN = ROOT / "Experiment/lighting_expansion_200/runs/lighting_200_visual_s17"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def plot_results(variants, history, selected_epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#67a9ff", "#c6a4ff", "#70e4bc", "#ffa66d"]
    names = ["Published baseline", "200 images", "300 support only", "300 detector + support"]
    with plt.rc_context({
        "figure.facecolor": "#111827", "axes.facecolor": "#172132",
        "axes.labelcolor": "#e6eefb", "axes.titlecolor": "#eef4ff",
        "axes.edgecolor": "#495970", "text.color": "#eef4ff",
        "xtick.color": "#c1cde1", "ytick.color": "#c1cde1",
        "font.size": 10, "grid.color": "#415069",
    }):
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), layout="constrained")
        fig.suptitle("Dance floor contact · 300-frame trial", fontsize=17, weight="bold")
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=.4, axis="y")
            axis.set_axisbelow(True)
        width = .19
        tolerances = (1, 2, 3, 5)
        for i, (scores, color, name) in enumerate(zip(variants.values(), colors, names, strict=True)):
            axes[0].bar([k + (i-1.5)*width for k in range(4)],
                        [scores["correct_by_tolerance_pct_width"][str(t)] for t in tolerances],
                        width=width, color=color, label=name)
            count = scores["hidden_measured"]
            axes[1].bar(i, count, color=color, width=.65)
            axes[1].text(i, count+.12, str(count), ha="center")
        axes[0].set(title="Visible contacts within tolerance", ylabel="Correct of 46",
                    xticks=range(4), xticklabels=[f"{t}%" for t in tolerances], ylim=(0, 46))
        axes[0].set_xlabel("Tolerance as a share of mat width")
        axes[0].legend(loc="upper left", fontsize=8, frameon=False)
        axes[1].set(title="Hidden contacts shown as measured", ylabel="Count of 50",
                    xticks=range(4), xticklabels=["Baseline", "200", "300 support", "300 joint"])
        axes[1].set_ylim(0, max(scores["hidden_measured"] for scores in variants.values()) + 2)
        epochs = [row["epoch"] for row in history]
        axes[2].plot(epochs, [row["train_loss"] for row in history], color=colors[0], label="Training")
        axes[2].plot(epochs, [row["validation"]["loss"] for row in history], color=colors[3], label="Validation")
        axes[2].axvline(selected_epoch, color=colors[2], linestyle="--", label=f"Selected epoch {selected_epoch}")
        axes[2].set(title="300-image support training", xlabel="Epoch", ylabel="Loss")
        axes[2].legend(loc="best", fontsize=8, frameon=False)
        fig.savefig(OUT / "results_comparison.png", dpi=160)
        plt.close(fig)


def main():
    qa = read(OUT / "structural_qa.json")
    support_dir = RUN / "evaluation"
    joint_dir = RUN / "joint_evaluation"
    support = read(support_dir / "results.json")
    joint = read(joint_dir / "results.json")
    detector = read(DETECTOR / "development_comparison.json")
    contact_training = read(RUN / "complete.json")
    detector_training = read(DETECTOR / "complete.json")
    support_weights = RUN / "best_joint.pt"
    detector_weights = DETECTOR / "weights/best.pt"
    assert sha(OUT / "training_manifest.json") == qa["training_manifest_sha256"]
    assert sha(support_weights) == support["selection"]["support_sha256"] == joint["selection"]["support_sha256"]
    assert sha(detector_weights) == joint["selection"]["detector_sha256"]
    assert qa["status"] == "passed"
    history = read(RUN / "history.json")
    selected_epoch = max(history, key=lambda row: row["validation"]["selected"]["joint_f1"])["epoch"]
    assert len(history) == 30
    variants = {
        "Published baseline": support["published_baseline"]["all"],
        "200-image revision": support["revision_200"]["all"],
        "300-image support with published detector": support["candidate_300_support"]["all"],
        "300-image support and detector": joint["candidate_300_joint"]["all"],
    }
    plot_results(variants, history, selected_epoch)
    score_rows = []
    for name, scores in variants.items():
        v, h = scores["visible_references"], scores["hidden_references"]
        counts = scores["correct_by_tolerance_pct_width"]
        assert (v, h) == (46, 50)
        score_rows.append([name, *(f"{counts[str(t)]}/{v}" for t in (1, 2, 3, 5)),
                           f"{scores['hidden_measured']}/{h}"])
    scores_table = text_table(score_rows, ["Pipeline", "Within 1%", "Within 2%", "Within 3%", "Within 5%", "Hidden reported measured"])
    detector_rows = []
    for name, scores in detector["models"].items():
        detector_rows.append(["Published detector" if name == "published_detector" else "300-image detector",
                              scores["predictions"], f"{scores['matched50']}/{scores['references']}",
                              f"{scores['matched75']}/{scores['references']}", f"{scores['f1']:.3f}"])
    detector_table = text_table(detector_rows, ["Model", "Predictions", "IoU ≥ 0.50", "IoU ≥ 0.75", "F1 at 0.50"])

    old_motion = read(OLD_RUN / "display_sweep.json")[-1]
    replays = {
        "200-image revision": old_motion,
        "300-image support with published detector": read(support_dir / "replay_coverage.json"),
        "300-image support and detector": read(joint_dir / "replay_coverage.json"),
    }
    replay_rows = []
    for name, row in replays.items():
        assert row["frames"] == 2424
        count = row["displayed_dots"]
        replay_rows.append([name, f"{count:,}", f"{count/row['frames']:.2f}",
                            row["single_frame_flickers"], f"{row['acceleration_p99']:.4f}"])
    replay_table = text_table(replay_rows, ["0.30 replay", "Map dots", "Dots per frame", "One-frame flickers", "Step-change p99"])

    report = f"""# Dance floor contact · 300-frame training results

The training set uses 300 fixed-camera source images and the original 11 validation images. A, B, and C contain 100 training images each. See [the dataset notes](DATASET.md) for review provenance, exact counts and instructions.

## Support and detector training

- The RTX 5080 trained the support head for 30 epochs in {contact_training['seconds']/60:.1f} minutes. It used ImageNet ResNet18 initialization, an attention head, posture-balanced sampling and image augmentations. Epoch {selected_epoch} was selected by validation joint F1. The deployed crop-averaged reportability threshold was {support['selection']['threshold']:.2f}.
- The RTX 5080 fine-tuned the dancer detector for {detector_training['epochs_requested']} epochs in {detector_training['seconds']/60:.1f} minutes using the existing detector checkpoint, the three overlapping crop regions, and the 300-image dataset. Ultralytics selected the checkpoint on validation. Annotation coverage in some dense frames remains incomplete.
- The previous tracker configuration, homography, and offline map fit were reused. The support-only comparison holds cached detections and tracks fixed. The joint comparison reruns the tracker with the new detector.

## Reserved floor contact comparison

{scores_table}

The seven reserved images are correlated samples from this choreography. The tolerance is measured as a percentage of mat width. Missing points count as misses. Hidden reports count solid measured dots for contacts that were hidden in the references. These comparisons use validation-selected thresholds. The reserved block had also been used in earlier trials. The reference labels are approximate and the comparison does not measure stable dancer identity.

![Confirmation scores and support training loss](results_comparison.png)

## Detector comparison on development images

{detector['scope']} Both detectors use the same tiled 960-pixel inference, confidence {detector['confidence']:.2f}, and merge IoU {detector['merge_iou']:.2f}.

{detector_table}

## 0.30 display replays

{replay_table}

Map dots include reported contacts and hollow temporal estimates. A flicker is a point appearing for just one frame or disappearing for just one frame. Step-change p99 measures the upper tail of frame-to-frame movement changes in units of 1% of mat width. These replay diagnostics include training frames and describe display behavior.

The results archive contains both 97-second replays and their map coordinates. Formation detection and dancer formation accuracy are separate project steps.
"""
    (OUT / "RESULTS.md").write_text(report, encoding="utf-8")
    names = {
        OUT / "RESULTS.md": "RESULTS.md",
        OUT / "DATASET.md": "DATASET.md",
        OUT / "REPRODUCE.md": "REPRODUCE.md",
        OUT / "results_comparison.png": "results_comparison.png",
        OUT / "detector_dev_references.json": "evaluation/detector_development_references.json",
        support_weights: "weights/support_300.pt",
        detector_weights: "weights/detector_300.pt",
        RUN / "history.json": "training/support_history.json",
        RUN / "complete.json": "training/support_complete.json",
        RUN / "settings.json": "training/support_settings.json",
        DETECTOR / "results.csv": "training/detector_history.csv",
        DETECTOR / "complete.json": "training/detector_complete.json",
        DETECTOR / "args.yaml": "training/detector_args.yaml",
        DETECTOR / "development_comparison.json": "evaluation/detector_development.json",
    }
    for directory, label in ((support_dir, "support_only"), (joint_dir, "joint")):
        for name in ("results.json", "selection.json", "development_calibration.json",
                     "replay_coverage.json", "map_positions.json", "lighting_trial_replay.mp4"):
            names[directory / name] = f"evaluation/{label}/{name}"
    for name in ("propose.py", "merge_reviewed.py", "verify_dataset.py", "train_300.py", "evaluate_300.py", "report_300.py"):
        names[OUT / name] = f"reproduction/{name}"
    for source in names:
        assert source.is_file(), source
    archive = OUT / "Dance_300_Results.zip"
    with ZipFile(archive, "w") as z:
        for path, name in names.items():
            z.write(path, name)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(names.values())
        for source, name in names.items():
            assert sha(source) == hashlib.sha256(z.read(name)).hexdigest()
    print("Verified results package:", archive.name)


if __name__ == "__main__":
    main()
