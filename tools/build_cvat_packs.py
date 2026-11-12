"""Build one CVAT image ZIP per reviewer from the shared lossless images."""
from pathlib import Path
from collections import Counter
import json, shutil, zipfile

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "Experiment/Dance_Overnight/CVAT_Review"
IMAGES = ROOT / "Experiment/Dance_Project_Shared/dataset/frames"

def build():
    seen = set()
    for reviewer in "ABC":
        source = REVIEW / f"teammate_{reviewer}"
        rows = json.loads((source / "manifest.json").read_text())
        names = [r["filename"] for r in rows]
        assert len(set(names)) == len(names) and not seen.intersection(names)
        seen.update(names)
        folder = ROOT / "build/cvat" / reviewer
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(folder / "images.zip", "w") as z:
            for name in names:
                path = IMAGES / name
                with path.open("rb") as image:
                    assert image.read(8) == b"\x89PNG\r\n\x1a\n", f"Missing PNG data: {name}. Run git lfs pull."
                z.write(path, name)
        for path in [source / "annotations.xml", REVIEW / "labels_schema.json"]:
            shutil.copy2(path, folder / path.name)
        counts = Counter(r["split"] for r in rows)
        print(f"{reviewer}: {counts['train']} train, {counts['val']} validation -> build/cvat/{reviewer}")

if __name__ == "__main__":
    build()

