"""Fresh MP4 to synchronized dancer video, square map, and reusable exports."""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

import cv2
import imageio_ffmpeg
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_dashboard_data import floor_guides, read, stage_projection
from project_external_replay import project
from register_external_stage import StageRegistration, register
from run_external_replay import infer, load_checkpoints

STAGE_METERS = 12
MAX_SECONDS = 180
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def inspect_video(source):
    if source.suffix.lower() != ".mp4":
        raise ValueError("Choose an MP4 video.")
    capture = cv2.VideoCapture(str(source))
    try:
        ok, image = capture.read()
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if not ok or frames < 2 or not 1 <= fps <= 120:
        raise ValueError("This MP4 could not be decoded or is too short.")
    height, width = image.shape[:2]
    if width * 9 != height * 16 or not 320 <= width <= 3840:
        raise ValueError("Use a 16:9 video between 320 and 3840 pixels wide.")
    if frames / fps > MAX_SECONDS:
        raise ValueError(f"Trim the clip to {MAX_SECONDS} seconds or less.")
    probe = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-xerror", "-i", str(source),
                            "-map", "0:v:0", "-vf", "vfrdet", "-an", "-f", "null", "-"],
                           capture_output=True, text=True, errors="replace")
    timing = re.search(r"VFR:[\d.]+ \((\d+)/(\d+)\)", probe.stderr)
    if probe.returncode or not timing:
        raise ValueError("FFmpeg found a decode or timestamp error in the MP4.")
    variable, constant = map(int, timing.groups())
    if variable > max(1, .01 * (variable + constant)):
        raise ValueError("Use a constant frame rate export so the video and stage map stay synchronized.")
    return frames, fps, [width, height]


def manual_calibration(corners, frames, fps, resolution):
    try:
        quad = np.asarray(corners, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Enter four numeric pixel coordinate pairs.") from error
    if quad.shape != (4, 2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
        raise ValueError("Enter four distinct corners in perimeter order.")
    if np.any(quad < 0) or np.any(quad >= resolution) or cv2.contourArea(quad, oriented=True) < 100:
        raise ValueError("Enter visible corners clockwise: back-left, back-right, front-right, front-left.")
    if not (quad[0, 0] < quad[1, 0] and quad[3, 0] < quad[2, 0]
            and quad[:2, 1].mean() < quad[2:, 1].mean()):
        raise ValueError("The corner order is back-left, back-right, front-right, front-left.")
    matrix = stage_projection(quad)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError("The floor corners do not define a valid planar map.")
    return {"source_resolution": resolution, "fps": fps, "frame_corners": [quad.tolist()] * frames,
            "calibrated_frames": frames, "method": "Manually entered corners for a fixed camera"}


def render(source, output, tracks, contacts, calibration, fps):
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(output / "intermediate.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1440, 614))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not open the video export encoder.")
    source_scale = 960 / calibration["source_resolution"][0]
    try:
        for frame, (boxes, points, corners) in enumerate(zip(tracks, contacts, calibration["frame_corners"], strict=True)):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Could not decode frame {frame}.")
            image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
            view = np.full((614, 1440, 3), (31, 23, 18), np.uint8)
            stage_left, stage_top, size = 983, 63, 432
            matrix = stage_projection(corners)
            px = lambda p: tuple(np.rint(np.asarray(p) * source_scale).astype(int))
            for label, a, b in floor_guides(matrix):
                cv2.line(image, px(a), px(b), (164, 136, 83), 1, cv2.LINE_AA)
            for row in boxes:
                if row["box"] is None:
                    continue
                a, b = px(row["box"][:2]), px(row["box"][2:])
                cv2.rectangle(image, a, b, (246, 189, 159), 1, cv2.LINE_AA)
                cv2.putText(image, str(row["id"]), (a[0], max(10, a[1]-3)), 0, .38, (255, 241, 214), 1, cv2.LINE_AA)
            for i in range(7):
                v = round(i * size / 6)
                cv2.line(view, (stage_left+v, stage_top), (stage_left+v, stage_top+size), (82, 66, 49), 1)
                cv2.line(view, (stage_left, stage_top+v), (stage_left+size, stage_top+v), (82, 66, 49), 1)
            for _, x, y, status, _ in points:
                u, v = x / 100, y / 62.5
                a, b, w = matrix @ [u, v, 1]
                where = (stage_left + round(size*u), stage_top + round(size*v))
                style = -1 if status == "proxy" else 1
                cv2.circle(image, px([a/w, b/w]), 5, (171, 231, 128), style, cv2.LINE_AA)
                cv2.circle(view, where, 5, (171, 231, 128), style, cv2.LINE_AA)
            view[45:585, :960] = image
            cv2.putText(view, f"Stage replay    Frame {frame}    {frame/fps:.2f} s", (16, 29), 0, .63, (241, 240, 232), 1, cv2.LINE_AA)
            cv2.putText(view, "12 x 12 m stage", (983, 29), 0, .63, (241, 240, 232), 1, cv2.LINE_AA)
            label = "Solid: reported   Hollow: estimated" if matrix is not None else "Stage alignment unavailable"
            cv2.putText(view, label, (983, 530), 0, .54, (241, 240, 232), 1, cv2.LINE_AA)
            writer.write(view)
    finally:
        writer.release()
        capture.release()
    target = output / "annotated_replay.mp4"
    encoded = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(output / "intermediate.mp4"),
                              "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
                              "-pix_fmt", "yuv420p", "-preset", "veryfast", "-c:a", "aac", "-t", str(len(tracks) / fps),
                              "-movflags", "+faststart", str(target)], capture_output=True, text=True)
    if encoded.returncode:
        raise RuntimeError(f"Could not encode the synchronized MP4: {encoded.stderr.strip()}")
    (output / "intermediate.mp4").unlink()
    check = cv2.VideoCapture(str(target))
    try:
        if int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != len(tracks) or abs(check.get(cv2.CAP_PROP_FPS)-fps) > .001:
            raise ValueError("Export timing differs from the source.")
    finally:
        check.release()
    return target


def run(source, output, device="auto", corners=None, progress=None):
    """Run fresh inference in an empty output directory and check export timing."""
    source, output = Path(source).resolve(), Path(output).resolve()
    start = perf_counter()
    frames, fps, resolution = inspect_video(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory for this run.")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select CPU to process the clip.")
    if progress:
        progress("Aligning stage", 0)
    if corners is None:
        # The input and all temporary files stay inside this run's folder.
        from shutil import copyfile
        staged = output / "source.mp4"
        copyfile(source, staged)
        register(staged, StageRegistration())
        calibration = read(staged.with_suffix(".calibration.json"))
        staged.unlink()
        staged.with_suffix(".calibration.json").unlink()
    else:
        calibration = manual_calibration(corners, frames, fps, resolution)
    if device != "cpu":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(int(device))
    detector, support, condition, selection = load_checkpoints(device)
    target = output / "inference.json"
    infer(source, target, detector, support, condition, selection, device,
          (lambda done, total: progress("Running dancer and contact models", done/total)) if progress else None)
    data = read(target)
    tracks, contacts = project(data, calibration, selection["display_threshold"], selection["settings"])
    if progress:
        progress("Rendering synchronized video", 1)
    render(source, output, tracks, contacts, calibration, fps)
    with (output / "contacts.csv").open("w", encoding="utf-8", newline="") as stream:
        csv_rows = csv.writer(stream)
        csv_rows.writerow(["frame", "time_s", "track_id", "x_m", "depth_m", "status", "age_frames"])
        for k, points in enumerate(contacts):
            for ident, x, y, status, age in points:
                csv_rows.writerow([k, k/fps, ident, round(x*.12, 4), round(y*12/62.5, 4),
                                   "reported" if status == "proxy" else "estimated", age])
    exports = {"fps": fps, "stage_m": [12, 12], "frames": [
        {"frame": k, "time_s": k/fps, "boxes": boxes, "contacts": [
            {"id": ident, "x_m": round(x*.12, 4), "depth_m": round(y*12/62.5, 4),
             "status": "reported" if status == "proxy" else "estimated", "age_frames": age}
            for ident, x, y, status, age in points]}
        for k, (boxes, points) in enumerate(zip(tracks, contacts, strict=True))]}
    elapsed = perf_counter()-start
    memory = psutil.Process().memory_info()
    metadata = {"frames": frames, "fps": fps, "resolution": resolution, "stage_m": [STAGE_METERS]*2,
                "device": data["device"], "inference_seconds": data["seconds"], "wall_seconds": elapsed,
                "inference_fps": frames / max(data["seconds"], .01),
                "process_peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(int(device))/2**20 if device != "cpu" else None,
                "frames_with_boxes": sum(any(row["box"] is not None for row in box) for box in tracks),
                "frames_with_contacts": sum(bool(frame) for frame in contacts),
                "calibrated_frames": calibration["calibrated_frames"], "checkpoint_config": data["config"],
                "contact_threshold": selection["display_threshold"], "trajectory_settings": selection["settings"],
                "processing": "Offline trajectory fitting uses future frames"}
    for name, value in (("calibration", calibration), ("metadata", metadata), ("frames", exports)):
        (output / f"{name}.json").write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    target.unlink()
    return metadata
