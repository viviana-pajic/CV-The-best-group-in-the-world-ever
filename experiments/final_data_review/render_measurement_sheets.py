"""Make raw image crops with a pixel coordinate scale and no labels."""
import csv
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "build/final_data_review"


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {int(row["source_frame"]): row for row in csv.DictReader(handle)}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    sample = rows(HERE / "sample.csv")
    targets = rows(HERE / "targets.csv")
    assert sample.keys() == targets.keys()
    for i in range(6):
        chunk = list(targets.items())[i * 6:(i + 1) * 6]
        sheet = Image.new("RGB", (1536, 968), "#162033")
        labels = ImageDraw.Draw(sheet)
        for j, (frame, target) in enumerate(chunk):
            ax, ay = int(target["x"]), int(target["y"])
            x0 = min(1040, max(0, (ax - 120) // 20 * 20))
            y0 = min(500, max(0, (ay - 140) // 20 * 20))
            tx, ty = (j % 3) * 512, (j // 3) * 484
            with Image.open(ROOT / sample[frame]["path"]) as source:
                crop = source.crop((x0, y0, x0 + 240, y0 + 220)).resize((480, 440)).convert("RGBA")
            marks = Image.new("RGBA", crop.size)
            draw = ImageDraw.Draw(marks)
            labels.text((tx + 34, ty + 2), f'{frame}   near ({ax}, {ay})', fill="white", font_size=17)
            for x in range(0, 240, 20):
                draw.line((x * 2, 0, x * 2, 440), fill=(180, 230, 255, 58))
                labels.text((tx + 34 + x * 2, ty + 24), str(x0 + x), fill="white", font_size=13)
            for y in range(0, 220, 20):
                draw.line((0, y * 2, 480, y * 2), fill=(180, 230, 255, 58))
                labels.text((tx + 1, ty + 44 + y * 2), str(y0 + y), fill="white", font_size=13)
            sheet.paste(Image.alpha_composite(crop, marks).convert("RGB"), (tx + 32, ty + 44))
        sheet.save(WORK / f"measure_{i + 1}.jpg", quality=94)


if __name__ == "__main__":
    main()
