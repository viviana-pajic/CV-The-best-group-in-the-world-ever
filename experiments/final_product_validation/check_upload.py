"""Run the upload page with a bundled short MP4 and inspect its output."""

import csv
import io
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import streamlit
import torch
from streamlit.testing.v1 import AppTest


def main():
    project = Path(sys.argv[1]).resolve()
    report = Path(sys.argv[2]).resolve()
    video = project / "experiments/integration/cpu_smoke/input.mp4"
    capture = cv2.VideoCapture(str(video))
    input_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    app = AppTest.from_file(project / "app/streamlit_app.py", default_timeout=120).run()
    assert not app.exception
    app.radio[0].set_value("Upload an MP4").run()
    assert app.button[0].disabled
    app.file_uploader[0].set_value((video.name, video.read_bytes(), "video/mp4")).run()
    app.radio[2].set_value("CPU").run()
    app.button[0].click().run(timeout=120)
    assert not app.exception and not app.error
    meta, files = app.session_state["result"]
    assert meta["device"] == "CPU" and meta["frames"] == input_frames
    assert meta["fps"] == fps and meta["calibrated_frames"] == input_frames
    assert meta["frames_with_contacts"] > 0
    expected = {
        "annotated_replay.mp4", "contacts.csv", "frames.json", "calibration.json", "metadata.json",
    }
    assert set(files) == expected and all(files.values())
    assert len(app.get("download_button")) == len(expected)
    frames = json.loads(files["frames.json"])["frames"]
    csv_rows = list(csv.DictReader(io.StringIO(files["contacts.csv"].decode("utf-8"))))
    assert len(frames) == input_frames and [f["frame"] for f in frames] == list(range(input_frames))
    assert len(csv_rows) == sum(len(f["contacts"]) for f in frames)
    assert all(abs(float(row["time_s"]) - int(row["frame"]) / fps) < 1e-9 for row in csv_rows)
    assert all(0 <= float(row[axis]) <= 12 for row in csv_rows for axis in ("x_m", "depth_m"))
    with TemporaryDirectory() as tmp:
        replay = Path(tmp) / "annotated_replay.mp4"
        replay.write_bytes(files[replay.name])
        output = cv2.VideoCapture(str(replay))
        try:
            assert output.get(cv2.CAP_PROP_FRAME_COUNT) == input_frames
            assert output.get(cv2.CAP_PROP_FPS) == fps
            assert output.read()[0]
            output.set(cv2.CAP_PROP_POS_FRAMES, input_frames - 1)
            assert output.read()[0]
        finally:
            output.release()

    app.file_uploader[0].set_value(("invalid.mp4", b"invalid MP4", "video/mp4")).run()
    assert not app.get("download_button")
    app.button[0].click().run()
    assert not app.exception and app.error and not app.get("download_button")
    invalid_upload_message = app.error[0].value
    app.radio[1].set_value("Enter four floor corners").run()
    app.text_area[0].set_value("null").run()
    app.button[0].click().run()
    assert not app.exception and app.error[0].value == "Enter four numeric pixel coordinate pairs."
    result = {
        "environment": {
            "python": platform.python_version(),
            "streamlit": streamlit.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "gpu_visible": torch.cuda.is_available(),
        },
        "device": meta["device"],
        "input": video.relative_to(project).as_posix(),
        "input_frames": input_frames,
        "fps": fps,
        "calibrated_frames": meta["calibrated_frames"],
        "frames_with_contacts": meta["frames_with_contacts"],
        "contact_rows": len(csv_rows),
        "elapsed_seconds": round(meta["wall_seconds"], 2),
        "download_files": sorted(files),
        "missing_upload_run_disabled": True,
        "output_video_first_and_last_decodable": True,
        "invalid_upload_message": invalid_upload_message,
        "old_downloads_visible_after_file_change": False,
        "invalid_manual_corners_message": app.error[0].value,
    }
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
