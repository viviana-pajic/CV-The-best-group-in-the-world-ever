"""Build shareable A/B/C and combined 300-frame annotation packs."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "Experiment/lighting_expansion_300"
PRIOR = ROOT / "Experiment/lighting_expansion_200/frames"
VALIDATION = ROOT / "Experiment/Dance_Project_Shared/dataset/frames"


def build_packs(output, batch):
    plan = json.loads((TRIAL / "assignments.json").read_text(encoding="utf-8"))
    checks = json.loads((TRIAL / "structural_qa.json").read_text(encoding="utf-8"))
    groups = plan["assignments"]
    new = set(plan["new_frames"])
    combined = sorted(frame for group in groups.values() for frame in group)
    if set(groups) != {"A", "B", "C"} or any(len(group) != 100 for group in groups.values()):
        raise ValueError("Expected three review batches of 100 frames.")
    if len(set(combined)) != 300:
        raise ValueError("Review batches must contain 300 distinct frames.")
    batches = [*groups, "combined"] if batch == "all" else [batch]
    output.mkdir(parents=True, exist_ok=True)
    for key in batches:
        frames = combined if key == "combined" else groups[key]
        xml = "merged_annotations.xml" if key == "combined" else f"{key}.xml"
        name = "Dance_300_Combined.zip" if key == "combined" else f"Batch_{key}_100.zip"
        path = output / name
        expected = checks["merged"] if key == "combined" else checks["parts"][key]
        xml_bytes = (TRIAL / xml).read_bytes()
        if hashlib.sha256(xml_bytes).hexdigest() != expected["xml_sha256"]:
            raise ValueError(f"{xml} differs from the verified export.")
        with NamedTemporaryFile(dir=output, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        try:
            with ZipFile(temporary, "w") as archive:
                archive.writestr("annotations.xml", xml_bytes, compress_type=ZIP_DEFLATED)
                archive.write(TRIAL / "DATASET.md", "DATASET.md", compress_type=ZIP_DEFLATED)
                for frame in frames:
                    filename = f"frame_{frame:06}.png"
                    folder = TRIAL / "new_frames" if frame in new else PRIOR
                    image = (folder / filename).read_bytes()
                    if hashlib.sha256(image).hexdigest() != checks["training_image_sha256"][str(frame)]:
                        raise ValueError(f"{filename} differs from the verified image.")
                    archive.writestr(f"images/{filename}", image)
                if key == "combined":
                    for document in ("assignments.json", "structural_qa.json", "training_manifest.json"):
                        archive.write(TRIAL / document, document, compress_type=ZIP_DEFLATED)
                    for file in sorted(TRIAL.glob("qa_*.json")):
                        archive.write(file, f"visual_review/{file.name}", compress_type=ZIP_DEFLATED)
                    for frame in plan["validation_frames"]:
                        filename = f"frame_{frame:06}.png"
                        archive.write(VALIDATION / filename, f"validation_images/{filename}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("A", "B", "C", "combined", "all"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "build/300-packs")
    options = parser.parse_args()
    build_packs(options.output, options.batch)
