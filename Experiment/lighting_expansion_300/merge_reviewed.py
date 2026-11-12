"""Merge the visual QA edits and produce the three review exports and training manifest."""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from annotator.annotation_io import Project, export_xml, import_xml, new_project
from propose import PREVIOUS, EXP, read, sheet, write
sys.path.insert(0, str(EXP / "code"))
from import_cvat_review import convert

import cv2


def fix_rows(rows, patch):
    result = deepcopy(rows)
    keyed = {row["id"]: row for row in result}
    for ident in patch.get("remove", []):
        del keyed[ident]
    for ident, fields in patch.get("updates", {}).items():
        row = keyed[int(ident)]
        row.update(fields)
        row["provenance"] = "Assistant visual correction against source frame."
    for added in patch.get("add", []):
        assert added["id"] not in keyed
        keyed[added["id"]] = {**added, "provenance": "Assistant visually added distinct dancer."}
    for row in keyed.values():
        row.update(sample_split="train", outside=False, reviewed=False)
        point = row["floor"]
        if point is not None:
            x, y = point
            a, b, c, d = row["box"]
            gap = max(0, a-x, b-y, x-c, y-d)
            assert gap <= 16, (row, "Review point far outside the body box")
            if gap:
                row["box"] = [max(0, min(a, x-1)), max(0, min(b, y-1)),
                              min(1280, max(c, x+1)), min(720, max(d, y+1))]
    return sorted(keyed.values(), key=lambda row: row["id"])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assignment = read(OUT / "assignments.json")
    new_ids = set(assignment["new_frames"])
    prior_ids = set(assignment["existing_frames"])
    reviewed, changes = set(), {}
    for name in ("qa_early", "qa_mid_1", "qa_mid_2", "qa_late"):
        data = read(OUT / f"{name}.json")
        frames = set(map(int, data["reviewed_frames"]))
        assert not reviewed & frames and frames <= new_ids
        assert set(map(int, data["changes"])) <= frames
        reviewed.update(frames)
        changes.update(data["changes"])
    assert reviewed == new_ids
    proposals = read(OUT / "new_proposals.json")
    additions = {k: fix_rows(rows, changes.get(k, {})) for k, rows in proposals.items()}
    assert set(map(int, additions)) == new_ids

    images = OUT / "frames"
    old = new_project(images, [(f"frame_{k:06}.png", 1280, 720) for k in sorted(prior_ids)])
    old = import_xml(old, PREVIOUS / "visual_relabel/verified_cvat_export.xml")
    floor = old.frames[0]["floor_guides"]
    frame_by_id = {int(f["name"][6:12]): f for f in old.frames}
    final_sheets = OUT / "qa_final"
    final_sheets.mkdir(exist_ok=True)
    for k, rows in additions.items():
        name = f"frame_{int(k):06}.png"
        dancers = [{"id": row["id"], "box": row["box"], "point": row["floor"],
                    "contact": row["contact"], "support_type": row["support_type"] or "none",
                    "sample_split": "train"} for row in rows]
        frame_by_id[int(k)] = {"name": name, "width": 1280, "height": 720, "dancers": dancers,
                               "floor_guides": deepcopy(floor)}
        sheet(cv2.imread(str(images / name)), rows, int(k), final_sheets / f"{int(k):06}.jpg")
    merged = OUT / "merged_annotations.xml"
    export_xml(Project(images, [frame_by_id[k] for k in sorted(frame_by_id)]), merged)
    for part, ids in assignment["assignments"].items():
        xml = OUT / f"{part}.xml"
        export_xml(Project(images, [frame_by_id[k] for k in ids]), xml)
        with zipfile.ZipFile(OUT / f"Batch_{part}_100.zip", "w") as bundle:
            bundle.write(xml, "annotations.xml")
            for k in ids:
                name = frame_by_id[k]["name"]
                bundle.write(images / name, f"images/{name}")

    training = convert(merged)
    prior = read(PREVIOUS / "visual_relabel/training_manifest.json")["lighting_200"]
    for k, rows in training.items():
        source = {row["id"]: row for row in additions[k] if int(k) in new_ids} if int(k) in new_ids else {
            row["id"]: row for row in prior["labels"][k]}
        for row in rows:
            row["reviewed"] = False
            row["provenance"] = source[row["id"]].get("provenance", "Existing 200-image visual revision.")
    valid = {str(k): prior["labels"][str(k)] for k in prior["val"]}
    training_manifest = {"lighting_300": {
        "source_first": 0, "train": sorted(frame_by_id), "val": prior["val"],
        "labels": training | valid,
        "provenance": {"merged_xml_sha256": sha(merged), "assignment_sha256": sha(OUT / "assignments.json"),
                       "scope": "200 existing assistant visually revised frames; 100 assisted proposals with assistant source-image visual review. A/B/C are review groups. No independent human verification."},
    }}
    write(OUT / "training_manifest.json", training_manifest)
    write(OUT / "new_visual_labels.json", additions)
    print("Merged", len(training), "frames; new contacts:", Counter(
        row["contact"] for rows in additions.values() for row in rows), flush=True)


if __name__ == "__main__":
    main()
