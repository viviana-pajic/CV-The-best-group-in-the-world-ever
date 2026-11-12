"""Check split, saved exports and upload input boundaries."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from src.pipeline import inspect_video, manual_calibration, run, stage_projection

ROOT = Path(__file__).resolve().parents[1]


class SubmissionChecks(unittest.TestCase):
    def test_disjoint_splits(self):
        exp = ROOT / "Experiment"
        trial = exp / "lighting_expansion_300"
        manifest = json.loads((trial / "training_manifest.json").read_text(encoding="utf-8"))["lighting_300"]
        train, val = set(manifest["train"]), set(manifest["val"])
        reserved = set(range(3938, 4093))
        self.assertEqual((len(train), len(val)), (300, 11))
        self.assertFalse(train & (val | reserved))
        self.assertGreater(min(abs(i-j) for i in train for j in val), 10)
        for rows in manifest["labels"].values():
            for row in rows:
                self.assertEqual(row["floor"] is not None, row["contact"] == "visible")

    def test_corners_project_and_invert(self):
        quad = [[80, 150], [200, 150], [300, 335], [20, 335]]
        calibration = manual_calibration(quad, 10, 25, [640, 360])
        H = stage_projection(calibration["frame_corners"][0])
        np.testing.assert_allclose(np.linalg.inv(H) @ H, np.eye(3), atol=1e-9)
        points = np.float32([[[0, 0], [1, 0], [1, 1], [0, 1]]])
        np.testing.assert_allclose(cv2.perspectiveTransform(points, H)[0], quad, atol=.001)
        for bad in (quad[::-1], [quad[i] for i in (0, 2, 1, 3)], [[80,150]]*4, {"wrong": "type"}):
            with self.assertRaises(ValueError):
                manual_calibration(bad, 10, 25, [640, 360])

    def test_rejects_invalid_video_and_existing_output(self):
        with TemporaryDirectory() as path:
            directory = Path(path)
            bad = directory / "broken.mp4"
            bad.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "decoded"):
                inspect_video(bad)
            with self.assertRaisesRegex(ValueError, "MP4"):
                inspect_video(directory / "other.txt")
            with self.assertRaisesRegex(ValueError, "empty output"):
                run(ROOT / "media/sample_static.mp4", directory, "cpu")

    def test_synchronized_saved_outputs(self):
        for folder in ("rtx_sample", "cpu_smoke"):
            directory = ROOT / "experiments/integration" / folder
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            frames = json.loads((directory / "frames.json").read_text(encoding="utf-8"))["frames"]
            self.assertEqual(len(frames), metadata["frames"])
            video = cv2.VideoCapture(str(directory / "annotated_replay.mp4"))
            try:
                self.assertEqual(video.get(cv2.CAP_PROP_FRAME_COUNT), len(frames))
                self.assertEqual(video.get(cv2.CAP_PROP_FPS), metadata["fps"])
                for k in (0, len(frames)-1):
                    video.set(cv2.CAP_PROP_POS_FRAMES, k)
                    self.assertTrue(video.read()[0])
            finally:
                video.release()
            with (directory / "contacts.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), sum(len(frame["contacts"]) for frame in frames))
            for row in rows:
                self.assertAlmostEqual(float(row["time_s"]), int(row["frame"])/metadata["fps"])
                self.assertIn(row["status"], {"reported", "estimated"})
                self.assertTrue(0 <= float(row["x_m"]) <= 12 and 0 <= float(row["depth_m"]) <= 12)


if __name__ == "__main__":
    unittest.main()
