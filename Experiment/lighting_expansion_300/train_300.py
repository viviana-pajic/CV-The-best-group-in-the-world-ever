"""Train support and detector candidates on the verified 300-image dataset."""
from collections import Counter
import argparse
import hashlib
import shutil
import sys
import time

import cv2
import torch
from ultralytics import YOLO

from propose import OUT, EXP, read, write
sys.path.insert(0, str(EXP / "code"))
import train_support
from detector_probe import TILES

MANIFEST = OUT / "training_manifest.json"
VAL_IMAGES = EXP.parent / "Dance_Project_Shared/dataset/frames"
DETECTOR = EXP / "detector_expanded/runs/diverse_brightness/weights/best.pt"
SUPPORT_NAME = "lighting_300_visual_s17"
DETECTOR_NAME = "detector_300_finetune_s17"


def path_for(frame, split):
    directory = OUT / "frames" if split == "train" else VAL_IMAGES
    return directory / f"frame_{frame:06}.png"


def train_contact(data):
    stage = OUT / "training_input"
    target = stage / "Dance_Project_Shared/dataset/frames"
    target.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for frame in data[split]:
            source = path_for(frame, split)
            path = target / source.name
            if not path.exists():
                try:
                    path.hardlink_to(source)
                except OSError:
                    shutil.copyfile(source, path)
    train_support.ROOT = stage
    train_support.OUT = OUT
    train_support.train(SUPPORT_NAME, MANIFEST, seed=17, wide=True, attention=True,
                        ignore_uncertain=True, weight_floor=30, epochs=30)


def tiled_dataset(data):
    root = OUT / "detector_data"
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for frame in data[split]:
            image = cv2.imread(str(path_for(frame, split)))
            assert image is not None, frame
            for tile, (l, t, r, b) in enumerate(TILES):
                width, height = r-l, b-t
                name = f"frame_{frame:06}_{tile}"
                lines = []
                for row in data["labels"][str(frame)]:
                    if row.get("outside") or row["box"] is None:
                        continue
                    x1, y1, x2, y2 = row["box"]
                    a, c, d, e = max(x1, l), max(y1, t), min(x2, r), min(y2, b)
                    if d-a < 6 or e-c < 8 or (d-a)*(e-c) < .1*(x2-x1)*(y2-y1):
                        continue
                    xywh = ((a+d-2*l)/(2*width), (c+e-2*t)/(2*height), (d-a)/width, (e-c)/height)
                    lines.append("0 " + " ".join(f"{v:.7f}" for v in xywh))
                assert cv2.imwrite(str(images / f"{name}.jpg"), image[t:b, l:r], [cv2.IMWRITE_JPEG_QUALITY, 97])
                (labels / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    path = root / "data.yaml"
    path.write_text(f"path: '{root.as_posix()}'\ntrain: images/train\nval: images/val\nnames: [dancer]\n",
                    encoding="utf-8")
    return path


def train_detector(data):
    if (OUT / "runs" / DETECTOR_NAME / "complete.json").is_file():
        print("Detector training is already complete.", flush=True)
        return
    path = tiled_dataset(data)
    start = time.perf_counter()
    model = YOLO(DETECTOR)
    result = model.train(data=str(path), imgsz=768, epochs=15, batch=6, workers=0, device=0, amp=False,
                         seed=17, deterministic=True, freeze=10, optimizer="AdamW", lr0=.00025,
                         patience=15, nbs=24, warmup_epochs=2, mosaic=0, mixup=0, copy_paste=0,
                         scale=.10, translate=.025, hsv_h=.005, hsv_s=.15, hsv_v=.15, fliplr=.5,
                         project=str(OUT / "runs"), name=DETECTOR_NAME, exist_ok=True, verbose=False)
    write(OUT / "runs" / DETECTOR_NAME / "complete.json",
          {"seconds": time.perf_counter()-start, "epochs_requested": 15, "training_images": 300,
           "validation_images": 11, "initialization": str(DETECTOR.relative_to(EXP)),
           "validation": result.results_dict, "best": str(model.trainer.best)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["support", "detector"])
    args = ap.parse_args()
    data = read(MANIFEST)["lighting_300"]
    audit = read(OUT / "structural_qa.json")
    assert audit["status"] == "passed"
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == audit["training_manifest_sha256"]
    print("CUDA:", torch.cuda.get_device_name(0), "runtime", torch.version.cuda,
          "training contact counts", Counter(r["contact"] for k in data["train"] for r in data["labels"][str(k)]),
          flush=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    {"support": train_contact, "detector": train_detector}[args.model](data)


if __name__ == "__main__":
    main()
