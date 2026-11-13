"""Select and package raw frames for the Formation Review training-label audit."""
import csv
import hashlib
from pathlib import Path
import statistics
import zipfile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PACK = ROOT / "data/final_annotation_review"
WORK = ROOT / "build/final_data_review"
SAMPLE = HERE / "sample.csv"
REVISED = "200-frame revision"
EXPANDED = "100-frame expansion"


def pick_sample(rows):
    """Pick 3 frames per batch per time quarter using image descriptors only."""
    selected = []
    frames = [int(row["source_frame"]) for row in rows]
    first, last = min(frames), max(frames)
    for batch in "ABC":
        for quarter in range(4):
            eligible = [row for row in rows if row["batch"] == batch and
                        4 * (int(row["source_frame"]) - first) // (last - first + 1) == quarter]
            center = statistics.median(float(row["mean_stage_luma"]) for row in eligible)
            for criterion, revision, score in [
                ("darker", REVISED if quarter % 2 == 0 else EXPANDED,
                 lambda row: float(row["mean_stage_luma"])),
                ("lower edge contrast", EXPANDED if quarter % 2 == 0 else REVISED,
                 lambda row: float(row["stage_laplacian_variance"])),
                ("typical lighting", REVISED,
                 lambda row: abs(float(row["mean_stage_luma"]) - center)),
            ]:
                pool = [row for row in eligible if row["revision"] == revision and
                        all(abs(int(row["source_frame"]) - int(old["source_frame"])) >= 18 for old in selected)]
                choice = min(pool, key=lambda row: (score(row), int(row["source_frame"])))
                selected.append(choice)
                choice["sample_reason"] = criterion
                choice["time_quarter"] = quarter + 1
    return sorted(selected, key=lambda row: int(row["source_frame"]))


def make_sheet(rows, index):
    sheet = Image.new("RGB", (1920, 1164), "#162033")
    draw = ImageDraw.Draw(sheet)
    for j, row in enumerate(rows):
        x, y = (j % 3) * 640, (j // 3) * 388
        draw.text((x + 12, y + 5), f'Frame {row["source_frame"]}   Batch {row["batch"]}', fill="white", font_size=19)
        with Image.open(ROOT / row["path"]) as im:
            sheet.paste(im.resize((640, 360)), (x, y + 28))
    sheet.save(WORK / f"raw_overview_{index}.jpg", quality=90)


def main():
    with (ROOT / "data/frame_manifest.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    selected = pick_sample(rows)
    assert len(selected) == len({row["source_frame"] for row in selected}) == 36
    assert {b: sum(row["batch"] == b for row in selected) for b in "ABC"} == dict.fromkeys("ABC", 12)
    fields = ["source_frame", "seconds", "batch", "revision", "time_quarter", "sample_reason",
              "mean_stage_luma", "stage_laplacian_variance", "sha256", "path"]
    with SAMPLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    PACK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACK / "raw_review_frames.zip", "w", compression=zipfile.ZIP_STORED) as z:
        z.write(SAMPLE, "sample.csv")
        z.write(PACK / "README.md", "README.md")
        for row in selected:
            frame = ROOT / row["path"]
            data = frame.read_bytes()
            assert hashlib.sha256(data).hexdigest() == row["sha256"], frame
            z.writestr(f"images/{frame.name}", data)
    WORK.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        make_sheet(selected[i * 9:(i + 1) * 9], i + 1)


if __name__ == "__main__":
    main()
