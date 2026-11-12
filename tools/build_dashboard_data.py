"""Package an existing tracked replay as a self-contained dashboard clip."""
import argparse
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = Path("lighting_expansion_300/runs/lighting_300_visual_s17/joint_evaluation")
ASSETS = ROOT / "dashboard/assets"
UNIT_CORNERS = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sample(video, index):
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = video.read()
    if not ok:
        raise ValueError(f"Could not decode frame {index}")
    return image


def stage_projection(corners):
    if corners is None:
        return None
    return cv2.getPerspectiveTransform(UNIT_CORNERS, np.float32(corners))


def project(inverse, x, y):
    u, v, scale = inverse @ [x, y, 1]
    return [round(float(u / scale), 1), round(float(v / scale), 1)]


def floor_guides(inverse):
    if inverse is None:
        return []
    lines = []
    for i in range(7):
        value = i / 6
        label = "mat_boundary" if i in (0, 6) else "floor_grid"
        lines.append([label, project(inverse, value, 0), project(inverse, value, 1)])
        lines.append([label, project(inverse, 0, value), project(inverse, 1, value)])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "media/source_choreography.mp4")
    parser.add_argument("--positions", type=Path, default=ROOT / "Experiment" / EVALUATION / "map_positions.json")
    parser.add_argument("--tracks", type=Path, default=ROOT / "build" / EVALUATION / "base_tracks.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "Experiment/Dance_CVAT_Pilot/calibration_draft.json")
    parser.add_argument("--guides", type=Path, default=ROOT / "Experiment/lighting_expansion_200/visual_relabel/floor_geometry.json")
    parser.add_argument("--id", default="stage-replay")
    parser.add_argument("--title", default="Original choreography")
    parser.add_argument("--stage-meters", type=float, nargs=2, metavar=("WIDTH", "DEPTH"),
                        help="Physical stage dimensions. The original replay defaults to 12 by 12 meters.")
    parser.add_argument("--start-frame", type=int, help="Preferred source frame to open on")
    parser.add_argument("--copy-video", action="store_true", help="Use the full original MP4 without re-encoding")
    args = parser.parse_args()
    if not args.id or not all(char.isascii() and (char.isalnum() or char in "-_") for char in args.id):
        parser.error("--id must use letters, digits, hyphens or underscores")
    stage_meters = args.stage_meters or ([12, 12] if args.id == "stage-replay" else None)
    if stage_meters and not all(isfinite(value) and value > 0 for value in stage_meters):
        parser.error("--stage-meters requires two positive finite dimensions")

    positions = read(args.positions)
    tracks = read(args.tracks)["frames"]
    calibration = read(args.calibration)
    frame_corners = calibration.get("frame_corners")
    guides = [] if frame_corners is not None else read(args.guides)["geometry"]
    count = len(positions["frames"])
    first, fps = positions["first_source_frame"], positions["fps"]
    if not count or count != len(tracks):
        raise ValueError("Tracks must cover the same frames as the exported contacts")
    if frame_corners is not None and len(frame_corners) != count:
        raise ValueError("Per-frame stage calibration must cover the same frames as the exported contacts")
    if positions["bounds"]["x"][0] or positions["bounds"]["y"][0]:
        raise ValueError("Expected a stage map with origin (0, 0)")
    width, depth = (positions["bounds"][axis][1] for axis in ("x", "y"))
    # Formation analysis uses a single scale: one unit equals 1% of stage width.
    stage = [100, 100 * stage_meters[1] / stage_meters[0] if stage_meters else 100 * depth / width]
    columns = [positions["fields"].index(key) for key in ("track_id", "x", "y", "map_status")]
    inverse = np.linalg.inv(calibration["pixel_to_relative_mat_homography"]) if frame_corners is None else None

    source = cv2.VideoCapture(str(args.source))
    resolution = (int(source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if tuple(calibration["source_resolution"]) != resolution or abs(source.get(cv2.CAP_PROP_FPS) - fps) > .001:
        raise ValueError("Source video must match the resolution and frame rate of the replay")
    preferred = args.start_frame if args.start_frame is not None else (4907 if args.id == "stage-replay" else first)
    start = min(count - 1, max(0, preferred - first))
    ASSETS.mkdir(parents=True, exist_ok=True)
    filename = f"{args.id}.mp4"
    poster = f"{args.id}.jpg"
    target = ASSETS / filename
    if args.copy_video:
        if first != 0 or count != int(source.get(cv2.CAP_PROP_FRAME_COUNT)) or args.source.suffix.lower() != ".mp4":
            raise ValueError("--copy-video requires a full-length MP4 replay")
        shutil.copyfile(args.source, target)
    else:
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(first / fps),
            "-i", str(args.source), "-t", str(count / fps), "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(target),
        ], check=True)
    replay = cv2.VideoCapture(str(target))
    if int(replay.get(cv2.CAP_PROP_FRAME_COUNT)) != count or abs(replay.get(cv2.CAP_PROP_FPS) - fps) > .001:
        raise ValueError("Encoded replay has an unexpected frame count or frame rate")
    for index in sorted({0, start, count // 2, count - 1}):
        expected = sample(source, first + index)
        actual = sample(replay, index)
        error = cv2.absdiff(expected, actual).mean()
        if error > 3:
            raise ValueError(f"Video alignment check failed at frame {index}: mean pixel difference {error:.2f}")
        print(f"Checked source frame {first + index}: mean pixel difference {error:.2f}")
    if not cv2.imwrite(str(ASSETS / poster), sample(replay, start)):
        raise ValueError("Could not write the replay poster")
    source.release()
    replay.release()

    frames = []
    for index, (contacts, rows) in enumerate(zip(positions["frames"], tracks, strict=True)):
        if frame_corners is not None:
            inverse = stage_projection(frame_corners[index])
        if contacts and inverse is None:
            raise ValueError(f"Uncalibrated frame {index} has map contacts")
        boxes = [
            [row["id"], *[round(float(value), 1) for value in row["box"]], int(row["status"] == "predicted")]
            for row in rows if row["status"] in {"observed", "predicted"} and row["box"] is not None
        ]
        points = []
        for contact in contacts:
            ident, x, y, status = (contact[column] for column in columns)
            if status not in {"proxy", "estimated"}:
                raise ValueError(f"Unexpected contact status {status!r}")
            relative_x, relative_y = x / width, y / depth
            points.append([ident, round(relative_x * stage[0], 3), round(relative_y * stage[1], 3),
                           int(status == "estimated"),
                           *project(inverse, relative_x, relative_y)])
        if not {point[0] for point in points}.issubset({row["id"] for row in rows}):
            raise ValueError("Map contacts and box tracks have inconsistent IDs")
        frame = [boxes, points]
        if frame_corners is not None:
            frame.append(floor_guides(inverse))
        frames.append(frame)

    clip = {
        "id": args.id, "title": args.title, "video": f"assets/{filename}", "poster": f"assets/{poster}",
        "firstFrame": first, "startFrame": start, "fps": fps, "threshold": positions["threshold"],
        "resolution": resolution, "stage": stage, "stageMeters": stage_meters, "guides": guides, "frames": frames,
    }
    path = ASSETS / f"{args.id}.js"
    payload = json.dumps(clip, separators=(",", ":"), allow_nan=False)
    path.write_text(f"(window.DANCE_CLIPS ??= []).push({payload});\n", encoding="utf-8")
    print(f"Exported {count} synchronized frames to {path.parent}")


if __name__ == "__main__":
    main()
