"""Extract the static-camera section or a requested frame range from the source."""
from pathlib import Path
import argparse
import cv2

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=3250)
    parser.add_argument("--last", type=int, default=5673)
    parser.add_argument("--output", type=Path, default=ROOT / "build/extracted_frames")
    a = parser.parse_args()
    if not 0 <= a.first <= a.last:
        parser.error("Use a nonnegative, inclusive frame range.")
    cap = cv2.VideoCapture(str(ROOT / "media/source_choreography.mp4"))
    if not cap.isOpened():
        raise OSError("Cannot open the source video. In a Git checkout, run git lfs pull.")
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        for i in range(a.last + 1):
            if i < a.first:
                if not cap.grab():
                    raise EOFError(f"Cannot read frame {i}.")
                continue
            ok, frame = cap.read()
            if not ok or not cv2.imwrite(str(a.output / f"frame_{i:06d}.png"), frame):
                raise OSError(f"Cannot extract frame {i}.")
    finally:
        cap.release()
    print(f"Extracted {a.last - a.first + 1} frames to {a.output}")

if __name__ == "__main__":
    main()
