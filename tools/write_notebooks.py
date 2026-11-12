from pathlib import Path
from nbformat import v4 as n
import nbformat

ROOT = Path(__file__).resolve().parents[1]
md, code = n.new_markdown_cell, n.new_code_cell
init = """from pathlib import Path
from collections import Counter
import json
from IPython.display import Image, Video, display

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / "tools/build_cvat_packs.py").exists()), None)
assert ROOT is not None, "Start Jupyter from the project folder."
EXP = ROOT / "Experiment/Dance_Overnight"
def read(name):
    return json.loads((EXP / name).read_text(encoding="utf-8"))
"""

def save(name, cells):
    nb = n.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    nbformat.write(nb, ROOT / name)

save("Labeling_Guide.ipynb", [
    md("# Labeling guide\n\nReview one CVAT pack per person. Use the frame filename to find examples."),
    code(init),
    md("### 1. Build and open your pack\n\nIn CVAT, create a standalone image task. Paste the contents of labels_schema.json into **Labels → Raw**, save, and upload images.zip. Open the task, then import annotations.xml as **CVAT for images 1.1**."),
    code("""import subprocess, sys
run = subprocess.run([sys.executable, str(ROOT / "tools/build_cvat_packs.py")], check=True, capture_output=True, text=True)
print(run.stdout, end="")
"""),
    md("### 2. Review each dancer\n\nKeep one tight box per distinguishable dancer. On the box, set contact and support_type. Add one floor_support point only for visible support. Link it with the same positive instance_id. For hidden, airborne or uncertain contact, remove the point and set support_type to none."),
    code("""display(Image(filename=str(EXP / "figures/label_examples.jpg"), width=1050))
"""),
    md("Place the point where the supporting sole, knee, seat or palm meets the floor. Use the midpoint of two clear supports. A raised ankle or the bottom edge of a box can be somewhere else."),
    code("""frames = read("CVAT_Review/review_assignments.json")
for reviewer, rows in frames.items():
    validation = [row["filename"] for row in rows if row["split"] == "val"]
    print(reviewer, "validation:", ", ".join(validation))
print("All frame filenames and priority checks: CVAT_Review/REVIEW_CHECKLIST.md")
"""),
    md("### 3. Start with these checks\n\n**A:** 003290 and 003640, overlapping boxes. **B:** 005371, ID 8, seated support. **C:** 004207, ID 5, planted foot; 005480, ID 1, seated support. Review the rest of the assigned images too. Preserve sample_split on every box."),
    code("""sys.path.insert(0, str(EXP / "code"))
from import_cvat_review import convert

# Replace this path with your own CVAT export to validate it.
xml = EXP / "CVAT_Review/teammate_A/annotations.xml"
checked = convert(xml)
print(f"Validated {len(checked)} images and {sum(map(len, checked.values()))} boxes.")
"""),
    md("Save and export **CVAT for images 1.1**. Send back the XML and reviewer name. The README shows how to validate and merge all three exports before retraining.")
])

save("Research_Notebook.ipynb", [
    md("# Research notebook · dancer floor positions\n\nGoal: find a stable floor point for each visible dancer in this video. Cells analyze saved experiment evidence. The README has the separate commands to rerun inference scoring and training."),
    code(init),
    md("### 1. Try a point derived from the box\n\nA box bottom is cheap. We also tried correcting it with box geometry alone."),
    code("""g = read("geometry/results.json")["chosen"]["validation"]["selected"]
v = read("ending_experiment/results.json")["reference"]["old_development"]["selected"]
for name, r in [("Box geometry", g), ("Image-based support", v)]:
    print(f"{name}: {r['correct']}/{r['refs']} within 3%; hidden reported visible: {r['hidden_emitted']}")
"""),
    md("The image-based model placed more points correctly and produced fewer false visible contacts. We kept the detector for boxes and trained a separate head for point and visibility."),
    md("### 2. Test smoothing on the same tracks"),
    code("""r = read("round2/previous_test_diagnostic.json")["continuity"]
for name, x in r.items():
    print(f"{name}: p99 acceleration {x['acceleration_p99']:.3f}; abrupt changes {x['abrupt_changes_over_half_pct']}")
"""),
    md("The offline fit reduced abrupt changes on this diagnostic replay. It uses future frames and can blur fast motion."),
    md("### 3. Try more data and heavier alternatives"),
    code("""m = read("data/development_ending.json")
for split in ("train", "val"):
    frames = [(d, k) for d in m.values() for k in d[split]]
    crops = [o for d, k in frames for o in d["labels"][str(k)] if o["box"] is not None and o["contact"] != "uncertain"]
    print(split, len(frames), "frames;", len(crops), "usable crops")

d = read("detector_expanded/tracking_selection.json")
for name in ("baseline", "chosen"):
    v = d[name]["validation"]
    print(f"Detector {name}: {v['correct_initial_ids']}/{v['reference_boxes']} matched the initial ID")
t = read("temporal_experiment/results.json")["runs"]
for name, r in t.items():
    print(name, r["controlled_development"]["correct_by_tolerance_pct_width"]["3"], "/ 77")
s = read("seed_repeatability/results.json")["models"]
for name, r in s.items():
    print(name, r["controlled"]["correct_by_tolerance_pct_width"]["3"], "/ 77")
"""),
    md("Extra posture examples stayed. The detector change gave a small gain in this ID check. Neighboring frames and averaging three models did not beat the selected single-image head on this development comparison."),
    md("### 4. Freeze the pipeline and score fresh frames\n\nCount every visible reference; a missing prediction is a miss. Distances are percentages of mat width."),
    code("""r = read("final_confirmation/results.json")
for name, model in r["candidates"].items():
    refs = [x for x in model["records"] if x["contact"] == "visible"]
    correct = sum(x["error_pct"] is not None and x["error_pct"] <= 3 for x in refs)
    hidden = [x for x in model["records"] if x["contact"] == "hidden"]
    false_visible = sum(x["state"] == 1 for x in hidden)
    assert correct == model["all"]["correct_by_tolerance_pct_width"]["3"]
    print(f"{name}: {correct}/{len(refs)} within 3%; {false_visible}/{len(hidden)} hidden marked visible")
"""),
    md("Keep **YOLO → tracking → learned support and visibility → offline smoothing**. The improvement comes from seven frames in one choreography with draft reference labels. Dense overlaps still lose positions and identities."),
    code("""display(Video(url="Experiment/Dance_Overnight/videos/final_confirmation_point_ablation.mp4", width=950))
"""),
    md("### 5. Try filling a few more gaps"),
    code("""d = read("weak_contact/results.json")["previous_confirmation_diagnostic"]
for name, r in d.items():
    a = r["all"]
    print(name, a["correct_by_tolerance_pct_width"]["3"], "/ 46; hidden estimates:", a["hidden_estimated"])
"""),
    md("The replay adds a few correct positions and more guesses for hidden dancers. Keep these hollow estimates optional; this is a replay of inspected frames."),
    md("Next: human review, retrain from the corrected labels, then write the separate end-to-end notebook once the final pipeline is settled.")
])

