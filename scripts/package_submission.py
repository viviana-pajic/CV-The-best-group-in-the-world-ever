"""Assemble a verified project folder, then archive its exact contents."""
import argparse
import json
from hashlib import file_digest
from pathlib import Path
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "build/300-packs"
TUNED = "Experiment/lighting_expansion_300/runs/detector_300_finetune_s17/weights/best.pt"
SUPPORT = "Experiment/lighting_expansion_300/runs/lighting_300_visual_s17/best_joint.pt"
PUBLISHED = "Experiment/Dance_Overnight/detector_expanded/runs/diverse_brightness/weights/best.pt"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def assemble(folder):
    folder.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    updated = [p for name in (".github", "app", "configs", "dashboard", "data", "docs", "experiments",
                               "models", "scripts", "src", "tests", "tools") for p in (ROOT/name).rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    files = {ROOT / p for p in tracked if p} | set(updated) | {ROOT / "media/sample_static.mp4", ROOT / "requirements-app.txt"}
    for path in sorted(files):
        copy(path, folder / path.relative_to(ROOT))

    for pack in sorted(PACKS.glob("Batch_?_100.zip")):
        with ZipFile(pack) as archive:
            if len([name for name in archive.namelist() if name.startswith("images/")]) != 100 or archive.testzip():
                raise ValueError(f"Invalid review pack: {pack.name}")
        copy(pack, folder / "data/review_packs" / pack.name)
    if len(list((folder / "data/review_packs").glob("Batch_?_100.zip"))) != 3:
        raise ValueError("Three review packs are required.")

    artifacts = ROOT / "build/lighting_expansion_300/runs/lighting_300_visual_s17"
    for name in ("evaluation", "joint_evaluation"):
        for filename in ("base_tracks.json", "predictions.json"):
            path = artifacts / name / filename
            if path.is_file():
                copy(path, folder / "experiments/300_frozen" / name / filename)
    for path in sorted((ROOT / "build/external-test-videos").glob("*.inference.json")):
        copy(path, folder / "experiments/replays/cache" / path.name)
    for path in sorted((ROOT / "build/external-test-videos").glob("*.tracks.json")):
        copy(path, folder / "experiments/replays/cache" / path.name)
    copy(ROOT / "build/check_external_dashboard.cjs", folder / "tests/check_dashboard.cjs")

    for label, source in {
        "stage_calibration.json": "Experiment/Dance_CVAT_Pilot/calibration_draft.json",
        "selected_pipeline.json": "Experiment/lighting_expansion_300/runs/lighting_300_visual_s17/joint_evaluation/selection.json",
        "detector_training.yaml": "experiments/detector_training/args.yaml",
        "support_training.json": "Experiment/lighting_expansion_300/runs/lighting_300_visual_s17/settings.json",
    }.items():
        copy(ROOT / source, folder / "configs" / label)

    weights = [{"role": role, "path": path, "sha256": digest(folder/path)} for role, path in (
        ("Generic COCO YOLO26m", "models/yolo26m.pt"), ("Earlier dancer detector", PUBLISHED),
        ("300-image dancer detector", TUNED), ("300-image floor-contact head", SUPPORT))]
    (folder / "models/checkpoints.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    history = subprocess.check_output(["git", "log", "--format=%h %as %s"], cwd=ROOT).decode("utf-8")
    (folder / "docs/git-history.txt").write_text(
        "Historical repository: https://github.com/GCM-Novosel/dance-floor-contact\n"
        "The submission folder includes the saved checkout and later local work.\n\n" + history, encoding="utf-8")
    print(f"Prepared {folder}", flush=True)


def archive(folder, zip_path):
    if zip_path.exists():
        raise FileExistsError(zip_path)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    manifest = folder / "SHA256SUMS.txt"
    files = [p for p in files if p != manifest]
    manifest.write_text("".join(f"{digest(path)}  {path.relative_to(folder).as_posix()}\n"
                                for path in files), encoding="utf-8")
    files.append(manifest)
    temporary = zip_path.with_suffix(".tmp")
    stored = {".mp4", ".png", ".jpg", ".jpeg", ".zip", ".exe", ".pt", ".pdf"}
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=4) as package:
        for path in files:
            package.write(path, path.relative_to(folder.parent).as_posix(),
                          compress_type=ZIP_STORED if path.suffix.lower() in stored else ZIP_DEFLATED)
    with ZipFile(temporary) as package:
        if package.testzip() is not None or len(package.namelist()) != len(files):
            raise ValueError("Archive verification failed")
    temporary.replace(zip_path)
    print(f"Verified {len(files):,} files, {zip_path.stat().st_size / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, help="Create a ZIP from an already prepared folder")
    args = parser.parse_args()
    if args.archive:
        archive(args.folder.resolve(), args.archive.resolve())
    else:
        assemble(args.folder.resolve())
