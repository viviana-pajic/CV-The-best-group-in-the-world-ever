"""Align the calibrated stage to new recordings of the same venue."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "media/source_choreography.mp4"
CALIBRATION = ROOT / "Experiment/Dance_CVAT_Pilot/calibration_draft.json"
WORK_SIZE = (1280, 720)
REFERENCE_FRAMES = (3250, 3600, 4254, 4670, 4907, 5125, 5400)
UNIT_CORNERS = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
STAGE_CHECK = np.float32([[314, 396], [987, 400], [200, 600], [1080, 604]])


def resize(image):
    return cv2.resize(image, WORK_SIZE, interpolation=cv2.INTER_AREA)


def read_frame(video, index):
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = video.read()
    if not ok:
        raise ValueError(f"Could not decode frame {index}")
    return resize(image)


class StageRegistration:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=1700, contrastThreshold=.019)
        self.mask = np.zeros(WORK_SIZE[::-1], np.uint8)
        # Exclude the performance area and keep the truss, banners and stage apron.
        self.mask[40:380, :] = 255
        self.mask[380:468, :155] = 255
        self.mask[380:468, 1125:] = 255
        self.mask[650:704, :] = 255
        cap = cv2.VideoCapture(str(REFERENCE))
        try:
            self.references = [self.features(read_frame(cap, i)) for i in REFERENCE_FRAMES]
        finally:
            cap.release()
        calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        self.reference_corners = np.float32(calibration["image_points"])

    def features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, self.mask)
        return np.float32([key.pt for key in keypoints]), descriptors

    def locate(self, image):
        points, descriptors = self.features(image)
        if descriptors is None or len(descriptors) < 32:
            return None
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=36))
        matcher.add([descriptors])
        matcher.train()
        best = 0
        corners = None
        for reference, reference_descriptors in self.references:
            matches = [pair[0] for pair in matcher.knnMatch(reference_descriptors, k=2)
                       if len(pair) == 2 and pair[0].distance < .73 * pair[1].distance]
            if len(matches) < max(30, best):
                continue
            src = np.float32([reference[match.queryIdx] for match in matches])
            dst = np.float32([points[match.trainIdx] for match in matches])
            transform, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 2.8)
            if transform is None:
                continue
            inliers = mask.ravel().astype(bool)
            count = int(inliers.sum())
            if count < 30 or count <= best or np.any(np.ptp(dst[inliers], axis=0) < [550, 115]):
                continue
            back_left, back_right, front_left, front_right = cv2.perspectiveTransform(
                STAGE_CHECK[None], transform)[0]
            width = back_right[0] - back_left[0]
            depth = (front_left[1] + front_right[1] - back_left[1] - back_right[1]) / 2
            if not (490 < width < 1040 and 140 < depth < 325 and 330 < back_left[1] < 475):
                continue
            projected = cv2.perspectiveTransform(self.reference_corners[None], transform)[0]
            if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
                continue
            best, corners = count, projected
        return corners


def distance(left, right):
    return np.max(np.linalg.norm(left - right, axis=1))


def interpolate(samples, count, fps):
    known = sorted((i, corners) for i, corners in samples.items() if corners is not None)
    frames = [None] * count
    if not known:
        return frames
    softened = []
    for k, (i, corners) in enumerate(known):
        if 0 < k < len(known) - 1:
            left_index, left = known[k - 1]
            right_index, right = known[k + 1]
            if i - left_index == right_index - i and max(distance(corners, left), distance(corners, right)) < 6:
                corners = (left + 2 * corners + right) / 4
        softened.append((i, corners))
    for i, corners in softened:
        frames[i] = corners
    for (a, left), (b, right) in zip(softened, softened[1:]):
        # Across a tracking failure, interpolate only when the framing still agrees.
        if b - a <= fps * .6 or (b - a <= fps * 2 and distance(left, right) <= 6):
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                frames[i] = left * (1 - t) + right * t
    first, last = softened[0][0], softened[-1][0]
    for i in range(max(0, first - round(fps * .3)), first):
        frames[i] = frames[first]
    for i in range(last + 1, min(count, last + round(fps * .3) + 1)):
        frames[i] = frames[last]
    return frames


def register(source, engine):
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resolution = [int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
    if not count or not fps or resolution[0] * 9 != resolution[1] * 16:
        raise ValueError(f"Expected a readable 16:9 source: {source}")
    step = max(1, round(fps / 3))
    samples = {}
    previous = None
    print(f"{source.stem}: registering stage across {count} frames", flush=True)
    try:
        for i in range(count):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Decode stopped at frame {i} in {source}")
            if i % step and i != count - 1:
                continue
            corners = engine.locate(resize(image))
            if previous is not None and corners is not None and distance(previous, corners) > 18:
                # Recheck large camera changes at full frame cadence.
                for index in range(max(0, i - step + 1), i):
                    samples[index] = engine.locate(read_frame(cap, index))
                cap.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
            previous = samples[i] = corners
    finally:
        cap.release()
    frames = interpolate(samples, count, fps)
    valid = sum(frame is not None for frame in frames)
    if valid < count / 2:
        raise RuntimeError(f"Could only align the stage in {valid} of {count} frames in {source}")
    scale = np.asarray(resolution) / WORK_SIZE
    result = {"source_resolution": resolution, "fps": fps,
              "method": "Stage landmarks registered to the original project calibration",
              "frame_corners": [(frame * scale).round(3).tolist() if frame is not None else None
                                for frame in frames],
              "matched_samples": sum(row is not None for row in samples.values()), "samples": len(samples),
              "calibrated_frames": valid}
    path = source.with_suffix(".calibration.json")
    path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"{source.stem}: stage aligned on {valid}/{count} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    engine = StageRegistration()
    for source in args.sources:
        register(source, engine)


if __name__ == "__main__":
    main()
