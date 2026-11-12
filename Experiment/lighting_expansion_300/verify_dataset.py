"""Validate the 300-frame annotation expansion before training."""

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path, PureWindowsPath
import re
import struct
import sys
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "Experiment"
PREVIOUS = EXPERIMENT / "lighting_expansion_200" / "visual_relabel"
CONFIRMATION = EXPERIMENT / "Dance_Overnight" / "data" / "final_confirmation"
HOLDOUT = set(range(3938, 4093))
FIXED_CAMERA = set(range(3250, 5674))
PNG = b"\x89PNG\r\n\x1a\n"
PARTS = ("A", "B", "C")
TOLERANCE = 0.02

sys.path.insert(0, str(ROOT))
from annotator.annotation_io import FLOOR_LABELS, import_xml, new_project

sys.path.insert(0, str(EXPERIMENT / "Dance_Overnight" / "code"))
from import_cvat_review import convert


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def frame_set(values, description):
    require(isinstance(values, list) and all(type(frame) is int for frame in values),
            f"{description}: expected a list of integer source frame numbers.")
    require(len(set(values)) == len(values), f"{description}: duplicate source frame numbers.")
    return set(values)


def xml_data(path):
    if path.suffix.lower() != ".zip":
        return path.read_bytes()
    with ZipFile(path) as archive:
        members = [entry for entry in archive.infolist() if PureWindowsPath(entry.filename).name == "annotations.xml"]
        require(len(members) == 1 and members[0].file_size <= 128 * 1024 * 1024,
                f"{path.name}: ZIP needs exactly one annotations.xml under 128 MiB.")
        return archive.read(members[0])


def verify_shape_tags(image, name):
    expected = {
        ("box", "dancer"): {"instance_id", "contact", "support_type", "sample_split"},
        ("points", "floor_support"): {"instance_id"},
        **{("polyline", label): set() for label in FLOOR_LABELS},
    }
    for shape in image:
        tag = (shape.tag, shape.get("label"))
        require(tag in expected, f"{name}: unexpected annotation shape {tag}.")
        attributes = [attribute.get("name") for attribute in shape.findall("attribute")]
        require(len(attributes) == len(set(attributes)) and set(attributes) == expected[tag],
                f"{name}: missing, duplicate, or unexpected attributes for {tag[1]}.")


def read_annotations(path, image_dir):
    data = xml_data(path)
    document = ET.fromstring(data)
    require(document.tag == "annotations" and document.find("track") is None,
            f"{path.name}: expected CVAT image annotations, without tracks.")
    descriptors, by_name, image_ids = [], {}, set()
    for image in document.findall("image"):
        name = PureWindowsPath(image.get("name", "")).name
        match = re.fullmatch(r"frame_(\d{6})\.png", name)
        require(match is not None, f"{path.name}: invalid training image name {name!r}.")
        frame = int(match[1])
        image_id = image.get("id")
        require(frame not in by_name and image_id not in image_ids and image_id is not None,
                f"{path.name}: duplicate frame or image ID at {name}.")
        image_ids.add(image_id)
        by_name[frame] = name
        width, height = int(image.attrib["width"]), int(image.attrib["height"])
        require((width, height) == (1280, 720), f"{name}: expected fixed-camera image size 1280×720.")
        verify_shape_tags(image, name)
        descriptors.append((name, width, height))

    project = import_xml(new_project(image_dir, descriptors), path)
    frames = dict(zip(by_name, project.frames))
    for frame, row in frames.items():
        require(bool(row["dancers"]), f"Frame {frame}: no dancer boxes.")
        require(all(dancer["sample_split"] == "train" for dancer in row["dancers"]),
                f"Frame {frame}: non-training row in training XML.")
    return frames, data


def normalized_guide(label, points):
    vertices = tuple(tuple(point) for point in points)
    return label, min(vertices, vertices[::-1])


def validate_guides(frames, reference):
    expected = Counter(normalized_guide(label, vertices) for label, *vertices in reference)
    for frame, row in frames.items():
        actual = Counter(normalized_guide(guide["label"], guide["points"]) for guide in row["floor_guides"])
        require(actual == expected, f"Frame {frame}: fixed-camera floor guide geometry differs.")


def verify_merged_rows(merged, reference, source, tolerance=TOLERANCE):
    for frame, row in reference.items():
        left = {dancer["id"]: dancer for dancer in row["dancers"]}
        right = {dancer["id"]: dancer for dancer in merged[frame]["dancers"]}
        require(left.keys() == right.keys(), f"{source}, frame {frame}: dancer IDs changed in merge.")
        for identity, expected in left.items():
            actual = right[identity]
            for field in ("contact", "support_type", "sample_split"):
                require(expected[field] == actual[field],
                        f"{source}, frame {frame}, dancer {identity}: {field} changed in merge.")
            for field in ("box", "point"):
                before, after = expected[field], actual[field]
                same = (
                    before is None and after is None
                    or before is not None and after is not None
                    and len(before) == len(after)
                    and all(abs(a - b) <= tolerance for a, b in zip(before, after))
                )
                require(same, f"{source}, frame {frame}, dancer {identity}: {field} changed in merge.")


def verify_images(frames, image_dir, previous_hashes):
    hashes = {}
    for frame, row in sorted(frames.items()):
        path = image_dir / row["name"]
        with path.open("rb") as stream:
            header = stream.read(24)
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        require(len(header) == 24 and header[:8] == PNG and header[12:16] == b"IHDR",
                f"{path.name}: invalid PNG header.")
        dimensions = struct.unpack(">II", header[16:24])
        require(dimensions == (row["width"], row["height"]), f"{path.name}: PNG size differs from XML.")
        if str(frame) in previous_hashes:
            require(digest == previous_hashes[str(frame)], f"{path.name}: existing training PNG changed.")
        hashes[str(frame)] = digest
    return hashes


def verify_manifest(path, merged_data, existing_manifest, train, validation):
    manifest = load_json(path)
    require(set(manifest) == {"lighting_300"}, "Training manifest should have one lighting_300 entry.")
    sample = manifest["lighting_300"]
    require(sample["source_first"] == 0, "Training manifest must use absolute source frame numbers.")
    require(frame_set(sample["train"], "Manifest training split") == train, "Training split does not match XML.")
    require(frame_set(sample["val"], "Manifest validation split") == validation, "Frozen validation split changed.")
    labels = sample["labels"]
    require(set(labels) == {str(frame) for frame in train | validation}, "Manifest label frames differ from split frames.")
    for frame in validation:
        key = str(frame)
        require(labels[key] == existing_manifest["labels"][key], f"Frozen validation labels changed in frame {frame}.")

    def without_provenance(rows):
        return sorted(
            ({key: value for key, value in row.items() if key not in {"reviewed", "provenance"}} for row in rows),
            key=lambda row: row["id"],
        )

    converted = convert(BytesIO(merged_data))
    for frame, rows in converted.items():
        require(without_provenance(rows) == without_provenance(labels[frame]),
                f"Frame {frame}: training rows differ from the project's CVAT converter.")


def stats(frames):
    contacts = Counter()
    supports = Counter()
    for row in frames.values():
        for dancer in row["dancers"]:
            contacts[dancer["contact"]] += 1
            if dancer["point"] is not None:
                supports[dancer["support_type"]] += 1
    return {
        "frames": len(frames),
        "dancer_boxes": sum(contacts.values()),
        "support_points": sum(supports.values()),
        "floor_polylines": sum(len(row["floor_guides"]) for row in frames.values()),
        "contacts": dict(sorted(contacts.items())),
        "support_types": dict(sorted(supports.items())),
        "frames_with_only_uncertain_contacts": [
            frame for frame, row in sorted(frames.items())
            if all(dancer["contact"] == "uncertain" for dancer in row["dancers"])
        ],
    }


def check_dataset(assignment_path, parts, merged_path, image_dir, training_manifest=None):
    require(set(parts) == set(PARTS), "Provide one XML or ZIP for each of A, B, and C.")
    assignment = load_json(assignment_path)
    groups = assignment["assignments"]
    require(set(groups) == set(PARTS), "Assignment manifest must have groups A, B, and C.")
    assigned = {part: frame_set(groups[part], f"Group {part}") for part in PARTS}
    new = frame_set(assignment["new_frames"], "New frames")
    old = frame_set(assignment["existing_frames"], "Existing frames")
    require(len(old) == 200 and len(new) == 100 and not old & new, "Expected 200 old and 100 distinct new frames.")
    train = old | new
    require(all(len(frames) == 100 for frames in assigned.values()) and set.union(*assigned.values()) == train,
            "A/B/C assignments must be disjoint groups of 100 covering all 300 training frames.")
    require(all(frames & old and frames & new for frames in assigned.values()),
            "Every A/B/C group must contain old and new frames.")

    previous_revision = load_json(PREVIOUS / "dataset_revision_manifest.json")
    previous_manifest = (PREVIOUS / "training_manifest.json").read_bytes()
    require(sha256(previous_manifest) == previous_revision["training_manifest_sha256"],
            "The frozen 200-frame training manifest changed.")
    existing = json.loads(previous_manifest)["lighting_200"]
    require(old == set(existing["train"]), "Existing frame list differs from the 200-frame baseline.")
    validation = set(existing["val"])
    require(len(validation) == 11, "Expected the original 11 validation frames.")
    confirmation_data = (CONFIRMATION / "labels.json").read_bytes()
    protocol = load_json(CONFIRMATION / "sealed_protocol.json")
    require(sha256(confirmation_data) == protocol["label_sha256"], "The sealed confirmation labels changed.")
    sealed = {int(frame) for frame in json.loads(confirmation_data)}
    require(train <= FIXED_CAMERA, "Training contains frames outside the fixed-camera segment.")
    require(not train & (HOLDOUT | sealed) and not validation & sealed,
            "Training or validation overlaps the sealed confirmation frames.")
    near_validation = {frame for frame in train if any(abs(frame - item) <= 10 for item in validation)}
    require(not near_validation, f"Training overlaps the ±10 validation guard: {sorted(near_validation)}.")

    guides = load_json(PREVIOUS / "floor_geometry.json")["geometry"]
    merged, merged_data = read_annotations(merged_path, image_dir)
    require(merged.keys() == train, "Merged XML does not contain exactly the 300 expected frames.")
    validate_guides(merged, guides)
    previous_hashes = previous_revision["image_sha256"]
    require(set(map(int, previous_hashes)) == old | validation,
            "Original image checksums must cover all old training and validation frames.")
    hashes = verify_images(merged, image_dir, previous_hashes)
    for frame in validation:
        image = EXPERIMENT / "Dance_Project_Shared" / "dataset" / "frames" / f"frame_{frame:06d}.png"
        require(sha256(image.read_bytes()) == previous_hashes[str(frame)], f"{image.name}: frozen validation PNG changed.")
    old_frames, old_data = read_annotations(PREVIOUS / "verified_cvat_export.xml", image_dir)
    require(sha256(old_data) == previous_revision["cvat_verification"]["export_sha256"],
            "The frozen 200-frame CVAT export changed.")
    require(old_frames.keys() == old, "Original 200 XML does not match the original manifest.")
    verify_merged_rows(merged, old_frames, "Existing XML", tolerance=0)

    part_report = {}
    for part, path in parts.items():
        frames, data = read_annotations(path, image_dir)
        require(frames.keys() == assigned[part], f"Group {part} XML differs from assigned frames.")
        validate_guides(frames, guides)
        verify_merged_rows(merged, frames, f"Group {part}")
        part_report[part] = {**stats(frames), "existing_frames": len(frames.keys() & old),
                             "new_frames": len(frames.keys() & new), "xml_sha256": sha256(data)}

    if training_manifest is not None:
        verify_manifest(training_manifest, merged_data, existing, train, validation)

    return {
        "status": "passed",
        "scope": "Structural, merge, and split verification. Annotation accuracy and reviewer identity require separate evidence.",
        "existing": stats(old_frames),
        "new": stats({frame: merged[frame] for frame in new}),
        "parts": part_report,
        "merged": {**stats(merged), "xml_sha256": sha256(merged_data)},
        "validation_frames": sorted(validation),
        "confirmation_excluded": [min(HOLDOUT), max(HOLDOUT)],
        "validation_guard_frames": 10,
        "merge_coordinate_tolerance_pixels": TOLERANCE,
        "png_dimensions_verified": len(merged),
        "baseline_png_hashes_verified": len(previous_hashes),
        "training_image_sha256": hashes,
        "training_manifest_sha256": sha256(training_manifest.read_bytes()) if training_manifest else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--part", action="append", required=True, metavar="A=PATH")
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    parts = {}
    try:
        for item in args.part:
            part, path = item.split("=", 1)
            require(part not in parts and bool(path), f"Duplicate or invalid part specification: {item}")
            parts[part] = Path(path)
        report = check_dataset(args.assignment_manifest, parts, args.merged, args.image_dir, args.training_manifest)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "training_image_sha256"}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError, BadZipFile) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")


if __name__ == "__main__":
    main()
