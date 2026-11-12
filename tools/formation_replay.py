"""Detect and score dancer lines in an offline floor-map replay."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import json
from math import atan, ceil, cos, degrees, isfinite, radians, sin, tan
from pathlib import Path
from statistics import mean, median
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "Experiment/lighting_expansion_300/runs/lighting_300_visual_s17/joint_evaluation"
MIN_WIDTH = 20.0
MAX_SEED_SPREAD = 8.0
MAX_SEED_ERROR = 2.6
MIN_SEED_GAP = 3.8
MIN_ROW_SEPARATION = 6.0


@dataclass(frozen=True)
class Candidate:
    ids: frozenset[int]
    center: float
    weight: float
    start: int
    stop: int
    angle: float = 0.0
    left: float = 0.0
    right: float = 100.0

    def depth_at(self, x):
        return self.center + (x - 50) * tan(radians(self.angle))


@dataclass
class Episode:
    identifier: int
    observations: list[tuple[int, Candidate]] = field(default_factory=list)


def find_candidates(points, minimum):
    """Find disjoint wide, narrow bands without specifying map positions."""
    ordered = sorted(points, key=lambda ident: points[ident][1])
    options = []
    solutions = []
    for start in range(len(ordered)):
        for stop in range(start + minimum, len(ordered) + 1):
            ids = ordered[start:stop]
            spread = points[ids[-1]][1] - points[ids[0]][1]
            if spread > MAX_SEED_SPREAD:
                break
            xs = sorted(points[ident][0] for ident in ids)
            width = xs[-1] - xs[0]
            if width < max(MIN_WIDTH, 3 * spread) or min(b - a for a, b in zip(xs, xs[1:])) < MIN_SEED_GAP:
                continue
            center = median(points[ident][1] for ident in ids)
            error = mean(abs(points[ident][1] - center) for ident in ids)
            if error > MAX_SEED_ERROR:
                continue
            weight = len(ids) - 1 + min(width, 50) / 50 - .3 * error
            option = Candidate(frozenset(ids), center, weight, start, stop)
            prior = [solution for other, solution in zip(options, solutions)
                     if other.stop <= start and center - other.center >= MIN_ROW_SEPARATION]
            total, rows = max(prior, key=lambda item: item[0], default=(0, []))
            options.append(option)
            solutions.append((total + weight, rows + [option]))
    return max(solutions, key=lambda item: item[0], default=(0, []))[1]


def robust_angle(coordinates, limit):
    """Estimate slope from well-separated pairs, then constrain the fitted angle."""
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in combinations(coordinates, 2)
              if abs(b[0] - a[0]) >= MIN_SEED_GAP]
    angle = degrees(atan(median(slopes))) if slopes else 0.0
    return min(limit, max(-limit, angle))


def line_coordinates(coordinates, angle):
    theta = radians(angle)
    dx, dy = cos(theta), sin(theta)
    along = [x * dx + y * dy for x, y in coordinates]
    normal = [y * dx - x * dy for x, y in coordinates]
    return along, normal


def compatible(first, second):
    if first.ids & second.ids:
        return False
    x = (max(first.left, second.left) + min(first.right, second.right)) / 2
    return (abs(first.angle - second.angle) > 8
            or abs(first.depth_at(x) - second.depth_at(x)) >= MIN_ROW_SEPARATION)


def find_angled_candidates(points, minimum, max_angle):
    """Search narrow strips in several directions, then refit their points continuously."""
    if len(points) < minimum:
        return []
    steps = max(1, ceil(max_angle / 3))
    angles = [i * max_angle / steps for i in range(-steps, steps + 1)]
    ids = list(points)
    coords = [points[ident] for ident in ids]
    seen = set()
    options = []
    for angle in angles:
        _, normal = line_coordinates(coords, angle)
        ordered = sorted(range(len(ids)), key=normal.__getitem__)
        for start in range(len(ids) - minimum + 1):
            for stop in range(start + minimum, len(ids) + 1):
                indices = ordered[start:stop]
                if normal[indices[-1]] - normal[indices[0]] > MAX_SEED_SPREAD:
                    break
                group = frozenset(ids[i] for i in indices)
                if group in seen:
                    continue
                seen.add(group)
                members = [points[ident] for ident in sorted(group)]
                fitted = robust_angle(members, max_angle)
                along, across = line_coordinates(members, fitted)
                positions = sorted(along)
                span = positions[-1] - positions[0]
                spread = max(across) - min(across)
                if (spread > MAX_SEED_SPREAD or span < max(MIN_WIDTH, 3 * spread)
                        or min(b - a for a, b in zip(positions, positions[1:])) < MIN_SEED_GAP):
                    continue
                offset = median(across)
                error = mean(abs(value - offset) for value in across)
                if error > MAX_SEED_ERROR:
                    continue
                theta = radians(fitted)
                center = (offset + 50 * sin(theta)) / cos(theta)
                xs = [point[0] for point in members]
                weight = len(group) - 1 + min(span, 50) / 50 - .3 * error
                options.append(Candidate(group, center, weight, 0, 0, fitted, min(xs), max(xs)))

    # Try several first choices so one long candidate cannot always crowd out two rows.
    ordered = sorted(options, key=lambda row: row.weight, reverse=True)
    solutions = []
    for seed in ordered[:12]:
        result = [seed]
        for option in ordered:
            if all(compatible(option, selected) for selected in result):
                result.append(option)
        solutions.append(result)
    return max(solutions, key=lambda rows: sum(row.weight for row in rows), default=[])


def group_episodes(proposals, max_gap):
    """Link consecutive row proposals by shared dancers and nearby depth."""
    episodes = []
    active = []
    for index, candidates in enumerate(proposals):
        active = [episode for episode in active if index - episode.observations[-1][0] <= max_gap]
        matches = []
        for episode in active:
            previous_index, previous = episode.observations[-1]
            for j, candidate in enumerate(candidates):
                common = len(previous.ids & candidate.ids)
                x = (max(previous.left, candidate.left) + min(previous.right, candidate.right)) / 2
                distance = abs(previous.depth_at(x) - candidate.depth_at(x))
                gap = index - previous_index
                if common >= 2 and distance <= 3.5 + .3 * gap and abs(previous.angle - candidate.angle) <= 9 + gap:
                    overlap = common / len(previous.ids | candidate.ids)
                    matches.append((overlap, -distance, episode.identifier, j))
        used_episodes, used_candidates = set(), set()
        for _, _, identifier, j in sorted(matches, reverse=True):
            if identifier not in used_episodes and j not in used_candidates:
                episodes[identifier].observations.append((index, candidates[j]))
                used_episodes.add(identifier)
                used_candidates.add(j)
        for j, candidate in enumerate(candidates):
            if j not in used_candidates:
                episode = Episode(len(episodes), [(index, candidate)])
                episodes.append(episode)
                active.append(episode)
    return episodes


def score_row(identifier, ids, points, zero_at):
    """Fit the row at its current median depth and measure perpendicular errors."""
    center = median(points[ident][1] for ident in ids)
    members = [
        {"track_id": ident, "x": points[ident][0], "y": points[ident][1],
         "offset": points[ident][1] - center}
        for ident in sorted(ids, key=lambda ident: points[ident][0])
    ]
    error = mean(abs(member["offset"]) for member in members)
    return {"line_id": identifier, "y": center,
            "left": members[0]["x"], "right": members[-1]["x"],
            "members": members, "mean_absolute_offset": error,
            "score": max(0.0, 100 * (1 - error / zero_at))}


def score_angled_row(identifier, ids, points, zero_at, limit, angle=None):
    ordered = sorted(ids, key=lambda ident: points[ident][0])
    coords = [points[ident] for ident in ordered]
    angle = robust_angle(coords, limit) if angle is None else min(limit, max(-limit, angle))
    along, across = line_coordinates(coords, angle)
    offset = median(across)
    theta = radians(angle)
    tangent = cos(theta), sin(theta)
    normal = -sin(theta), cos(theta)
    errors = [value - offset for value in across]
    projections = [[x - error * normal[0], y - error * normal[1]]
                   for (x, y), error in zip(coords, errors)]
    members = [{"track_id": ident, "x": point[0], "y": point[1],
                "offset": error, "projection": projection}
               for ident, point, error, projection in zip(ordered, coords, errors, projections)]
    error = mean(abs(value) for value in errors)
    start, end = [[position * tx + offset * nx for tx, nx in zip(tangent, normal)]
                  for position in (min(along), max(along))]
    return {"line_id": identifier, "y": (offset + 50 * sin(theta)) / cos(theta),
            "left": start[0], "right": end[0], "start": start, "end": end,
            "angle_degrees": angle, "members": members, "mean_absolute_offset": error,
            "score": max(0.0, 100 * (1 - error / zero_at))}


def smooth_angles(frames, measured, zero_at, max_angle):
    """Stabilize each tracked line's angle over five frames of offline playback."""
    angles = {(index, row["line_id"]): row["angle_degrees"]
              for index, frame in enumerate(frames) for row in frame["rows"]}
    for index, frame in enumerate(frames):
        rows = []
        for row in frame["rows"]:
            if row["mean_absolute_offset"] < 1e-9:
                rows.append(row)
                continue
            identifier = row["line_id"]
            angle = row["angle_degrees"]
            weighted = 3 * angle
            total = 3
            for direction in (-1, 1):
                for distance in (1, 2):
                    other = angles.get((index + direction * distance, identifier))
                    if other is None:
                        break
                    weight = 3 - distance
                    weighted += weight * other
                    total += weight
            ids = [member["track_id"] for member in row["members"]]
            rows.append(score_angled_row(identifier, ids, measured[index], zero_at, max_angle,
                                         angle=weighted / total))
        frame["rows"] = rows
        frame["overall"] = mean(row["score"] for row in rows) if rows else None


def analyze(data, minimum=4, zero_at=6.0, max_angle=0.0):
    fps = data["fps"]
    fields = [data["fields"].index(name) for name in ("track_id", "x", "y", "map_status")]
    id_col, x_col, y_col, status_col = fields
    measured = [
        {row[id_col]: (row[x_col], row[y_col]) for row in frame if row[status_col] == "proxy"}
        for frame in data["frames"]
    ]
    proposals = ([find_angled_candidates(points, minimum, max_angle) for points in measured]
                 if max_angle else [find_candidates(points, minimum) for points in measured])
    minimum_observations = max(3, ceil(.48 * fps))
    episodes = group_episodes(proposals, max(2, round(.28 * fps)))
    templates = defaultdict(list)
    vote_radius = max(2, round(.48 * fps))
    for episode in episodes:
        indices = [index for index, _ in episode.observations]
        if len(indices) < minimum_observations or len(indices) / (indices[-1] - indices[0] + 1) < .65:
            continue
        for index in range(indices[0], indices[-1] + 1):
            left = bisect_left(indices, index - vote_radius)
            right = bisect_right(indices, index + vote_radius)
            nearby = [candidate for _, candidate in episode.observations[left:right]]
            center = median(candidate.center for candidate in nearby)
            angle = median(candidate.angle for candidate in nearby)
            counts = Counter(ident for candidate in nearby for ident in candidate.ids)
            votes = {ident: count / len(nearby) for ident, count in counts.items()
                     if count >= ceil(.40 * len(nearby))}
            templates[index].append((episode.identifier, center, tan(radians(angle)), votes))

    result = []
    for index, points in enumerate(measured):
        groups = defaultdict(list)
        claims = {}
        for identifier, center, slope, votes in templates[index]:
            for ident in votes.keys() & points.keys():
                x, y = points[ident]
                priority = (votes[ident], -abs(y - center - (x - 50) * slope), -identifier)
                if ident not in claims or priority > claims[ident][0]:
                    claims[ident] = priority, identifier
        for ident, (_, identifier) in claims.items():
            groups[identifier].append(ident)
        rows = []
        for identifier, ids in groups.items():
            if len(ids) >= 3:
                if max_angle:
                    row = score_angled_row(identifier, ids, points, zero_at, max_angle)
                    width = (row["right"] - row["left"]) / cos(radians(row["angle_degrees"]))
                else:
                    row = score_row(identifier, ids, points, zero_at)
                    width = row["right"] - row["left"]
                if width >= MIN_WIDTH:
                    rows.append(row)
        # Remove duplicate lines after recovering their local dancer assignments.
        distinct = []
        selected = []
        for row in sorted(rows, key=lambda r: (len(r["members"]), r["score"]), reverse=True):
            candidate = Candidate(frozenset(member["track_id"] for member in row["members"]),
                                  row["y"], 0, 0, 0, row.get("angle_degrees", 0),
                                  row["left"], row["right"])
            if all(compatible(candidate, other) for other in selected):
                distinct.append(row)
                selected.append(candidate)
        distinct.sort(key=lambda row: row["y"])
        overall = mean(row["score"] for row in distinct) if distinct else None
        result.append({"source_frame": data["first_source_frame"] + index,
                       "overall": overall, "rows": distinct})
    if max_angle:
        smooth_angles(result, measured, zero_at, max_angle)
    return result


def draw_frame(source, dots, result, fps, index, max_angle=0):
    import cv2
    import numpy as np

    bg = (33, 24, 19)
    panel = (47, 36, 28)
    grid = (69, 56, 46)
    ink = (249, 245, 237)
    muted = (193, 174, 157)
    palette = [(204, 237, 118), (243, 182, 193), (119, 198, 255), (250, 204, 124), (169, 155, 250)]
    canvas = np.full((900, 1600, 3), bg, np.uint8)

    def label(value, x, y, size=.52, color=ink, weight=1):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, weight, cv2.LINE_AA)

    def rect(x, y, w, h, color):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)

    def line(a, b, color, weight=1):
        cv2.line(canvas, a, b, color, weight, cv2.LINE_AA)

    def center_text(value, x, y, size, color):
        width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0]
        label(value, x - width // 2, y, size, color)

    elapsed = index / fps
    title = f"Formation alignment  |  +/-{max_angle:g} deg" if max_angle else "Horizontal formation alignment"
    label(title, 24, 52, 1.14, ink, 2)
    label(f"300-image replay   |   {int(elapsed // 60):02}:{elapsed % 60:05.2f}   |   Source frame {result['source_frame']}",
          26, 88, .58, muted)
    canvas[140:635, 24:904] = source[120:615, 22:902]
    label("CHOREOGRAPHY", 24, 127, .47, muted)
    rect(24, 653, 879, 166, panel)
    label("How to read the map", 43, 685, .65, ink, 2)
    label("Colored lines follow the position and angle of their dots." if max_angle else
          "Colored lines show the fitted depth of each detected row.", 43, 720, .53, muted)
    label("Short marks show perpendicular distance to the line." if max_angle else
          "Short vertical marks show how far each dancer is off the line.", 43, 750, .53, muted)
    label("Hollow dots are estimated positions. Scored dancers are solid.", 43, 780, .53, muted)

    rows = result["rows"]
    headline = f"{result['overall']:.1f}%" if rows else "--"
    label(headline, 932, 69, 1.3, palette[0] if rows else muted, 2)
    label("OVERALL ALIGNMENT", 1125, 52, .55, ink, 1)
    noun = "line" if max_angle else "row"
    label(f"{len(rows)} detected {noun}{'' if len(rows) == 1 else 's'}", 1125, 80, .55, muted)
    map_x, map_y, map_w, map_h = 930, 147, 640, 400

    def pixel(x, y):
        return round(map_x + map_w * x / 100), round(map_y + map_h * y / 62.5)

    rect(map_x, map_y, map_w, map_h, panel)
    for step in range(9):
        x = map_x + round(map_w * step / 8)
        line((x, map_y), (x, map_y + map_h), grid)
    for step in range(7):
        y = map_y + round(map_h * step / 6)
        line((map_x, y), (map_x + map_w, y), grid)
    center_text("BACK", map_x + map_w // 2, map_y - 11, .43, muted)
    center_text("FRONT", map_x + map_w // 2, map_y + map_h + 19, .43, muted)
    def map_line(a, b, color, weight=1):
        visible, start, end = cv2.clipLine((map_x, map_y, map_w + 1, map_h + 1), pixel(*a), pixel(*b))
        if visible:
            line(start, end, color, weight)

    member_colors = {}
    for number, row in enumerate(rows):
        color = palette[number % len(palette)]
        left = row.get("start", [row["left"], row["y"]])
        right = row.get("end", [row["right"], row["y"]])
        map_line(left, right, color, 2)
        for member in row["members"]:
            member_colors[member["track_id"]] = color
            projection = member.get("projection", [member["x"], row["y"]])
            map_line(projection, (member["x"], member["y"]), color)
    for ident, x, y, status, _ in dots:
        point = pixel(x, y)
        color = member_colors.get(ident, (161, 143, 125))
        cv2.circle(canvas, point, 7, bg, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 5, color, -1 if status == "proxy" else 1, cv2.LINE_AA)
        name = str(ident)
        offset = 9 if point[0] < map_x + map_w - 28 else -23
        label(name, point[0] + offset, point[1] + 4, .35, color)

    label(noun.upper(), 948, 600, .43, muted)
    label("SCORED", 1040 if max_angle else 1072, 600, .43, muted)
    if max_angle:
        label("ANGLE", 1135, 600, .43, muted)
    label("MEAN OFFSET", 1240 if max_angle else 1182, 600, .43, muted)
    label("ALIGNMENT", 1435, 600, .43, muted)
    if not rows:
        rect(930, 614, 640, 58, panel)
        label(f"No sustained {noun} detected" if max_angle else "No sustained horizontal row detected",
              949, 650, .58, muted)
    for number, row in enumerate(rows):
        y = 613 + number * 42
        color = palette[number % len(palette)]
        rect(930, y, 640, 38, panel)
        rect(930, y, 4, 38, color)
        label(f"{noun.title()} {number + 1}", 948, y + 25, .54, color, 1)
        label(str(len(row["members"])), 1058 if max_angle else 1090, y + 25, .54, ink)
        if max_angle:
            label(f"{row['angle_degrees']:+.1f} deg", 1135, y + 25, .47, ink)
        label(f"{row['mean_absolute_offset']:.2f}% of mat width", 1240 if max_angle else 1182, y + 25, .47, muted)
        label(f"{row['score']:.1f}%", 1452, y + 25, .59, color, 2)
    return canvas


def render(data, result, replay, output, max_angle=0):
    import cv2
    import imageio_ffmpeg

    capture = cv2.VideoCapture(str(replay))
    fps = data["fps"]
    try:
        if not capture.isOpened() or capture.get(cv2.CAP_PROP_FRAME_COUNT) != len(result) or capture.get(cv2.CAP_PROP_FPS) != fps:
            raise ValueError("The input replay does not match the positions file.")
        def preview_rank(index):
            frame = result[index]
            rows = frame["rows"]
            return (sum(abs(row.get("angle_degrees", 0)) >= 12 for row in rows),
                    len(rows),
                    sum(len(row["members"]) for row in rows), frame["overall"] or 0)

        preview = max(range(len(result)), key=preview_rank)
        stem = "angled_formations" if max_angle else "horizontal_formations"
        with TemporaryDirectory(dir=output) as directory:
            raw = Path(directory) / "preview.mp4"
            final = Path(directory) / f"{stem}.mp4"
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1600, 900))
            if not writer.isOpened():
                raise RuntimeError("Could not open the video writer.")
            try:
                for index, (dots, metrics) in enumerate(zip(data["frames"], result, strict=True)):
                    ok, source = capture.read()
                    if not ok or source.shape[:2] != (760, 1600):
                        raise ValueError(f"Unexpected replay frame size or end of video at frame {index}.")
                    image = draw_frame(source, dots, metrics, fps, index, max_angle)
                    writer.write(image)
                    if index == preview:
                        cv2.imwrite(str(output / f"{stem}_preview.jpg"), image)
            finally:
                writer.release()
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(final)], check=True)
            final.replace(output / f"{stem}.mp4")
    finally:
        capture.release()
    return result[preview]["source_frame"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, default=DEMO / "map_positions.json")
    parser.add_argument("--replay", type=Path, default=DEMO / "lighting_trial_replay.mp4")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-angle", type=float, default=0.0,
                        help="Maximum absolute angle in degrees from horizontal (default: 0)")
    parser.add_argument("--zero-at", type=float, default=6.0, help="Mean offset in percent of mat width giving a 0%% score")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if not isfinite(args.max_angle) or not 0 <= args.max_angle <= 45:
        parser.error("--max-angle must be between 0 and 45 degrees")
    if not isfinite(args.zero_at) or args.zero_at <= 0:
        parser.error("--zero-at must be positive")
    data = json.loads(args.positions.read_text(encoding="utf-8"))
    expected = ["track_id", "x", "y", "map_status", "map_age_frames"]
    if data["fields"] != expected or data["bounds"] != {"x": [0, 100], "y": [0, 62.5]}:
        raise ValueError("The positions file must use the published floor-map coordinate format.")
    if not data["frames"]:
        raise ValueError("The positions file has no frames.")
    result = analyze(data, zero_at=args.zero_at, max_angle=args.max_angle)
    output = args.output or ROOT / "build" / ("angled-formations" if args.max_angle else "horizontal-formations")
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter(len(frame["rows"]) for frame in result)
    summary = {"fps": data["fps"], "frames": len(result), "frames_by_row_count": dict(sorted(counts.items())),
               "zero_score_mean_offset_pct_mat_width": args.zero_at,
               "measured_contacts_only": True, "max_angle_degrees": args.max_angle,
               "method": "Temporal dancer grouping, constrained robust line fits, five-frame offline angle smoothing, and perpendicular offsets."
               if args.max_angle else "Horizontal rows fitted at the current median depth of temporally associated measured dots."}
    if not args.no_video:
        summary["preview_source_frame"] = render(data, result, args.replay, output, args.max_angle)
    (output / "formation_scores.json").write_text(
        json.dumps({"summary": summary, "frames": result}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
