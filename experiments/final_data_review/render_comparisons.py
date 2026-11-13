"""Render side-by-side evidence for selected review disagreements."""
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "build/final_data_review"
BLIND = "#62e8d6"
PRIOR = "#ffc15e"


def draw_panel(sheet, tile, frame, record):
    x, y = (tile % 3) * 400, (tile // 3) * 344
    boxes = [box for box in (record["blind_box"], record.get("prior_box")) if box]
    center_x, center_y, width = 714, 490, 200
    if boxes:
        a, b = min(box[0] for box in boxes), min(box[1] for box in boxes)
        c, d = max(box[2] for box in boxes), max(box[3] for box in boxes)
        center_x, center_y = round((a+c)/2), round((b+d)/2)
        width = math.ceil(max(200, c-a+30, (d-b+30)*10/7))
    height = round(width*.7)
    left = min(1280-width, max(0, center_x-width//2))
    top = min(720-height, max(0, center_y-height//2))
    sx, sy = 400/width, 280/height
    with Image.open(ROOT / frame["path"]) as source:
        crop = source.crop((left, top, left + width, top + height)).resize((400, 280))
    draw = ImageDraw.Draw(crop)
    for box, point, color in (
        (record.get("prior_box"), record.get("prior_point"), PRIOR),
        (record["blind_box"], record["blind_point"], BLIND),
    ):
        if box:
            draw.rectangle(tuple(round((value-offset)*scale) for value, offset, scale in
                                 zip(box, (left, top, left, top), (sx, sy, sx, sy))),
                           outline=color, width=3)
        if point:
            px, py = round((point[0]-left)*sx), round((point[1]-top)*sy)
            draw.ellipse((px-5, py-5, px+5, py+5), fill=color, outline="black", width=1)
    sheet.paste(crop, (x, y + 64))
    labels = ImageDraw.Draw(sheet)
    distance = record.get("point_distance_px")
    diagnostic = f"{distance:.1f} px" if distance is not None and record["match"] == "matched" else record["match"].replace("_", " ")
    labels.text((x + 12, y + 6), f'Frame {record["source_frame"]}  ·  {diagnostic}', fill="white", font_size=18)
    labels.text((x + 12, y + 35), f'Blind  {record["blind_contact"]}', fill=BLIND, font_size=14)
    labels.text((x + 190, y + 35), f'Existing  {record.get("prior_contact", "n/a")}', fill=PRIOR, font_size=14)


def make_sheet(frames, records, chosen, dest):
    sheet = Image.new("RGB", (1200, 688), "#162033")
    for tile, frame in enumerate(chosen):
        draw_panel(sheet, tile, frames[frame], records[frame])
    sheet.save(dest, quality=92)


def main():
    with (HERE / "sample.csv").open(newline="", encoding="utf-8") as handle:
        frames = {int(row["source_frame"]): row for row in csv.DictReader(handle)}
    data = json.loads((HERE / "comparison.json").read_text(encoding="utf-8"))
    records = {row["source_frame"]: row for row in data["per_frame"]}
    WORK.mkdir(parents=True, exist_ok=True)
    make_sheet(frames, records, [3468, 3733, 4096, 4261, 4980, 5422], WORK / "comparison_1.jpg")
    make_sheet(frames, records, [5473, 5524, 5626, 5651, 5670, 5131], WORK / "comparison_2.jpg")
    make_sheet(frames, records, [3468, 3733, 4096, 4980, 5422, 5524], HERE / "review_examples.jpg")


if __name__ == "__main__":
    main()
