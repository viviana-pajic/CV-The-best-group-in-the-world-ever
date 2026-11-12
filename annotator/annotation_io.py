"""Read and write annotation drafts and CVAT image annotations."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile


CONTACTS = ("uncertain", "visible", "hidden", "airborne")
SUPPORT_TYPES = ("none", "feet", "knees", "mixed_knee_foot", "seat", "palms")
SPLITS = ("train", "val", "reserved_test")
FLOOR_LABELS = ("mat_boundary", "mat_rear_estimate", "mat_grid")
DRAFT_VERSION = 1


@dataclass
class Project:
    image_dir: Path
    frames: list[dict]
    last_image: str | None = None


def new_project(image_dir, frames):
    project = Project(Path(image_dir).resolve(), [
        {"name": name, "width": width, "height": height,
         "dancers": [], "floor_guides": []}
        for name, width, height in frames
    ])
    _validate(project)
    return project


def load_project(path):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or data.get("version") != DRAFT_VERSION:
            raise ValueError("This annotation project uses an unsupported file version.")
        image_dir = Path(data["image_dir"])
        if not image_dir.is_absolute():
            image_dir = (path.parent / image_dir).resolve()
        project = Project(image_dir, data["frames"], data.get("last_image"))
        _validate(project)
    except (KeyError, TypeError) as exc:
        raise ValueError("This file is missing required annotation project data.") from exc
    return project


def save_project(project, path):
    _validate(project)
    path = Path(path)
    image_dir = project.image_dir.resolve()
    try:
        location = os.path.relpath(image_dir, path.parent.resolve())
    except ValueError:
        location = str(image_dir)  # A relative path cannot cross Windows drives.
    data = {"version": DRAFT_VERSION, "image_dir": location,
            "frames": project.frames, "last_image": project.last_image}
    _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))


def _atomic_write(path, data):
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _coordinates(values, count):
    return (
        isinstance(values, (list, tuple))
        and len(values) == count
        and all(type(value) in (int, float) and math.isfinite(value) for value in values)
    )


def _in_image(point, width, height):
    return _coordinates(point, 2) and 0 <= point[0] < width and 0 <= point[1] < height


def _validate(project):
    if not isinstance(project.frames, list) or not project.frames:
        raise ValueError("An annotation project must contain at least one image.")
    seen_names = set()
    for frame in project.frames:
        name, width, height = frame["name"], frame["width"], frame["height"]
        if not isinstance(name, str) or name in ("", ".", "..") or name != PureWindowsPath(name).name:
            raise ValueError("Project images must use plain filenames.")
        if name.casefold() in seen_names:
            raise ValueError(f"Duplicate image name: {name}")
        seen_names.add(name.casefold())
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError(f"{name}: invalid image size.")
        if not isinstance(frame["dancers"], list) or not isinstance(frame["floor_guides"], list):
            raise ValueError(f"{name}: invalid annotation lists.")
        if frame.get("default_split", "train") not in SPLITS:
            raise ValueError(f"{name}: invalid default sample split.")
        ids = set()
        for row in frame["dancers"]:
            dancer_id = row["id"]
            message = f"{name}, dancer {dancer_id}"
            if type(dancer_id) is not int or dancer_id < 1 or dancer_id in ids:
                raise ValueError(f"{message}: dancer IDs must be positive and unique per image.")
            ids.add(dancer_id)
            box = row["box"]
            if not _coordinates(box, 4) or not (
                0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
            ):
                raise ValueError(f"{message}: box must have positive area inside the image.")
            contact, point = row["contact"], row["point"]
            support = row["support_type"]
            if contact not in CONTACTS:
                raise ValueError(f"{message}: invalid contact status.")
            if contact == "visible":
                if not _in_image(point, width, height) or support not in SUPPORT_TYPES[1:]:
                    raise ValueError(f"{message}: visible contact needs one support point and a support type.")
            elif point is not None or support not in (None, "none"):
                raise ValueError(f"{message}: only visible contact can have a support point or type.")
            if row.get("sample_split", "train") not in SPLITS:
                raise ValueError(f"{message}: invalid sample split.")
        for guide in frame["floor_guides"]:
            vertices = guide["points"]
            if guide["label"] not in FLOOR_LABELS or len(vertices) < 2 or not all(
                _in_image(vertex, width, height) for vertex in vertices
            ):
                raise ValueError(f"{name}: floor guide needs at least two points inside the image.")


def _shape_attributes(shape):
    return {item.attrib.get("name"): item.text or "" for item in shape.findall("attribute")}


def _shape_id(shape, attributes):
    raw = attributes.get("instance_id") or attributes.get("dancer_id") or shape.get("group_id")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dancer boxes and support points need a positive instance_id.") from exc


def _parse_points(shape):
    return [[float(value) for value in pair.split(",")] for pair in shape.attrib["points"].split(";")]


def import_xml(project, path):
    """Return a new project with annotations replaced on XML images only."""
    path = Path(path)
    try:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                entries = [item for item in archive.infolist()
                           if PureWindowsPath(item.filename).name == "annotations.xml"]
                if len(entries) != 1 or entries[0].file_size > 128 * 1024 * 1024:
                    raise ValueError("Choose a CVAT ZIP containing one annotations.xml under 128 MiB.")
                with archive.open(entries[0]) as source:
                    document = ET.parse(source).getroot()
        else:
            document = ET.parse(path).getroot()
    except (ET.ParseError, BadZipFile, RuntimeError) as exc:
        raise ValueError(f"Could not read the CVAT annotations: {exc}") from exc
    images = document.findall("image")
    if document.tag != "annotations" or not images:
        raise ValueError("Select a CVAT image annotations XML containing at least one image.")
    result = deepcopy(project)
    frames = {frame["name"].casefold(): frame for frame in result.frames}
    imported = set()
    for image in images:
        name = PureWindowsPath(image.get("name", "")).name
        key = name.casefold()
        if key not in frames:
            raise ValueError(f"{name}: image is missing from the selected folder.")
        if key in imported:
            raise ValueError(f"{name}: duplicate image in XML.")
        imported.add(key)
        frame = frames[key]
        try:
            if (int(image.attrib["width"]), int(image.attrib["height"])) != (frame["width"], frame["height"]):
                raise ValueError("XML dimensions do not match the selected image.")
            _import_frame(frame, image)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"{name}: {exc}") from exc
    _validate(result)
    return result


def _import_frame(frame, image):
    dancers, points, floor_guides = {}, {}, []
    for shape in image:
        label = shape.get("label")
        if shape.tag == "box" and label == "dancer":
            attributes = _shape_attributes(shape)
            dancer_id = _shape_id(shape, attributes)
            if dancer_id in dancers:
                raise ValueError(f"Duplicate box for dancer {dancer_id}.")
            dancers[dancer_id] = {
                "id": dancer_id,
                "box": [float(shape.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr")],
                "contact": attributes.get("contact", "uncertain"),
                "support_type": attributes.get("support_type") or "none",
                "sample_split": attributes.get("sample_split", "train"),
                "point": None,
            }
        elif shape.tag == "points" and label == "floor_support":
            dancer_id = _shape_id(shape, _shape_attributes(shape))
            vertices = _parse_points(shape)
            if dancer_id in points or len(vertices) != 1:
                raise ValueError(f"Dancer {dancer_id} must have at most one support point.")
            points[dancer_id] = vertices[0]
        elif shape.tag == "polyline" and label in FLOOR_LABELS:
            floor_guides.append({"label": label, "points": _parse_points(shape)})
    for dancer_id, point in points.items():
        if dancer_id not in dancers:
            raise ValueError(f"Support point has no box for dancer {dancer_id}.")
        dancers[dancer_id]["point"] = point
    frame["dancers"] = list(dancers.values())
    frame["floor_guides"] = floor_guides


def _field(parent, name, value):
    ET.SubElement(parent, name).text = str(value)


def _schema(parent):
    instance = ("instance_id", "text", ("0",))
    attributes = (
        instance, ("contact", "select", CONTACTS), ("support_type", "select", SUPPORT_TYPES),
        ("sample_split", "select", ("unset", *SPLITS)),
    )
    definitions = (
        ("dancer", "rectangle", "#253f46", attributes),
        ("floor_support", "points", "#b298eb", (instance,)),
        ("mat_boundary", "polyline", "#e7a652", ()),
        ("mat_rear_estimate", "polyline", "#e08200", ()),
        ("mat_grid", "polyline", "#f41c06", ()),
    )
    labels = ET.SubElement(parent, "labels")
    for name, shape_type, color, spec in definitions:
        label = ET.SubElement(labels, "label")
        _field(label, "name", name)
        _field(label, "type", shape_type)
        _field(label, "color", color)
        fields = ET.SubElement(label, "attributes")
        for field, input_type, values in spec:
            attribute = ET.SubElement(fields, "attribute")
            _field(attribute, "name", field)
            _field(attribute, "mutable", "True")
            _field(attribute, "input_type", input_type)
            _field(attribute, "default_value", values[0])
            _field(attribute, "values", "\n".join(values))


def _make_shape(image, shape, label, geometry, attributes=None):
    element = ET.SubElement(image, shape, {
        "label": label, "source": "manual", "occluded": "0", "z_order": "0",
        **{key: str(value) for key, value in geometry.items()},
    })
    for key, value in (attributes or {}).items():
        ET.SubElement(element, "attribute", name=key).text = str(value)


def _point_text(vertices):
    return ";".join(f"{x},{y}" for x, y in vertices)


def export_xml(project, path):
    """Write CVAT 1.1 image XML with the same labels used by the project importer."""
    _validate(project)
    root = ET.Element("annotations")
    _field(root, "version", "1.1")
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    _field(task, "name", project.image_dir.name)
    _field(task, "size", len(project.frames))
    _field(task, "mode", "annotation")
    _field(task, "start_frame", 0)
    _field(task, "stop_frame", len(project.frames) - 1)
    _schema(task)
    for index, frame in enumerate(project.frames):
        image = ET.SubElement(root, "image", id=str(index), name=frame["name"],
                              width=str(frame["width"]), height=str(frame["height"]))
        for row in frame["dancers"]:
            _make_shape(image, "box", "dancer", dict(zip(("xtl", "ytl", "xbr", "ybr"), row["box"])), {
                "instance_id": row["id"], "contact": row["contact"],
                "support_type": row["support_type"] or "none",
                "sample_split": row.get("sample_split", "train"),
            })
            if row["point"] is not None:
                _make_shape(image, "points", "floor_support", {"points": _point_text([row["point"]])},
                            {"instance_id": row["id"]})
        for guide in frame["floor_guides"]:
            _make_shape(image, "polyline", guide["label"], {"points": _point_text(guide["points"])})
    ET.indent(root, space="  ")
    _atomic_write(Path(path), ET.tostring(root, encoding="utf-8", xml_declaration=True))
