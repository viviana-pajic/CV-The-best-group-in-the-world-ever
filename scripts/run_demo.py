"""Process an MP4 with the same entry point used by the upload app."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=ROOT / "media/sample_static.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "build/fresh_demo")
    parser.add_argument("--device", choices=("auto", "cpu", "0"), default="auto")
    parser.add_argument("--corners", type=Path, help="JSON list of four ordered [x, y] pixel coordinates")
    args = parser.parse_args()
    corners = json.loads(args.corners.read_text(encoding="utf-8")) if args.corners else None
    try:
        result = run(args.source, args.output, args.device, corners)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
