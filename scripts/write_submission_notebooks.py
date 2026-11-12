"""Refresh the defense walkthrough and append the submission evidence to research."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
SECTION = "## 10. Evaluate boxes with standard AP"


def md(text):
    return nb.v4.new_markdown_cell(text.strip())


def code(text):
    return nb.v4.new_code_cell(text.strip())


setup = code(r"""
from pathlib import Path
from csv import DictReader
import json
from IPython.display import Image, Markdown, Video, display

root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'tools/run_notebooks.py').is_file())
exp = root / 'Experiment'
early = exp / 'Dance_Overnight'
trial = exp / 'lighting_expansion_300'
evidence = root / 'experiments'
run = trial / 'runs/lighting_300_visual_s17'

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]
    display(Markdown('\n'.join(lines)))

def show(path, width=930):
    display(Image(filename=str(path), width=width))
""")

ap_cell = code(r"""
report = load(evidence / 'detector/comparison.json')
names = {'generic': 'COCO YOLO26m', 'published': 'Earlier dance detector', 'tuned_300': '300-image detector'}
def ap_row(key):
    dev = report['models'][key]['development']
    reserved = report['models'][key]['reserved_diagnostic']
    return [names[key], f"{dev['mAP50']:.3f}", f"{dev['mAP50_95']:.3f}",
            f"{reserved['mAP50']:.3f}", f"{reserved['mAP50_95']:.3f}"]
table(['Detector', 'Dev AP50', 'Dev AP50:95', 'Reserved AP50', 'Reserved AP50:95'],
      [ap_row(key) for key in names])
""")

contact_cell = code(r"""
joint = load(run / 'joint_evaluation/results.json')
support = load(run / 'evaluation/results.json')
models = [
    ('Published baseline', joint['published_baseline']['all']),
    ('200-image support revision', joint['revision_200']['all']),
    ('300 support with earlier detector', support['candidate_300_support']['all']),
    ('300 support and detector', joint['candidate_300_joint']['all']),
]
table(['Pipeline', 'Within 1%', 'Within 2%', 'Within 3%', 'Within 5%', 'Hidden reported'],
      [[name, *[f"{value['correct_by_tolerance_pct_width'][k]}/46" for k in ('1','2','3','5')],
        f"{value['hidden_measured']}/50"] for name, value in models])
""")

master = [
    md("""
# Dance formation and floor contact

We built an offline tool to inspect where dancers touch the floor and how they line up. The original fixed-camera choreography is our development source. Wicked Games, onare and Dernek are new-performance visual checks. The team roadmap records verbal approval to use dance for the challenge.

[Interactive replay](dashboard/index.html) · [Research evidence](Research_Notebook.ipynb) · [Labeling guide](Labeling_Guide.ipynb) · [Rubric and remaining work](docs/rubric_and_gaps.md)

The cells below read saved experiment evidence and run on CPU. [Run a fresh clip](README.md) to execute the detector, tracker, support model, map and export from an MP4.
"""),
    setup,
    md("""
## 1. Find dancers, then find their floor contact

Generic YOLO finds many standing dancers. Overlapping crops help keep small performers large enough for inference, then cross-crop IoU suppression merges duplicates. Tracking links detections across frames. A box bottom was a useful first baseline, but floor work often puts the actual support at a knee, seat or palm. The learned head uses a body crop to estimate a point and whether the frame provides enough evidence to report it.

[Tiled detector](Experiment/Dance_Overnight/code/detector_probe.py) · [Contact model and crop averaging](Experiment/Dance_Overnight/code/candidate_evaluate.py)
"""),
    code(r"""
geometry = load(early / 'geometry/results.json')['chosen']['validation']['selected']
learned = load(early / 'ending_experiment/results.json')['reference']['old_development']['selected']
table(['Development comparison', 'Within 3% of width', 'Hidden reported'],
      [[name, f"{row['correct']}/{row['refs']}", row['hidden_emitted']]
       for name, row in [('Box geometry', geometry), ('Learned contact', learned)]])
show(early / 'figures/label_examples.jpg')
"""),
    md("""
## 2. Label the difficult examples

The pilot used 60 training frames and 11 validation frames. The 200-frame revision added lighting and floor-work variation, then the final run added another 100. We corrected proposals for cut-off boxes, duplicates, missed dancers and misplaced contacts. Only a visible contact gets a point. Hidden, airborne and uncertain remain separate label states.

A/B/C are review batches. The saved work documents model-assisted proposals and assistant visual review. The blind repeat was by the assistant and is evidence of repeatability within that workflow. Independent agreement still requires separate team labels.
"""),
    code(r"""
audit = load(trial / 'structural_qa.json')
split = load(evidence / 'data/split_audit.json')
table(['Pack', 'Frames', 'Boxes', 'Visible contact points'],
      [[name, row['frames'], row['dancer_boxes'], row['support_points']] for name, row in audit['parts'].items()])
print(f"{split['unique_verified_training_files']} verified unique training files; "
      f"{split['minimum_train_validation_gap_frames']} frames at the closest train/validation boundary.")
print(f"Reserved exclusion: {split['reserved_excluded_inclusive']}. Same performance across splits.")
"""),
    md("""
## 3. Fine-tune and compare the box detectors

The 300-frame detector starts from the earlier dance checkpoint and fine-tunes on tiled dancer boxes. It freezes the first ten layers, uses AdamW with learning rate 0.00025, and keeps mosaic off. For the comparison below we reran all three checkpoints with the same tiles, 960-pixel inference and merge IoU. AP integrates the precision–recall curve at an IoU threshold. AP50:95 averages ten IoU thresholds, so it penalizes loose boxes more.

The 300-frame model leads development AP50. Generic YOLO leads AP50:95 on both available diagnostic splits. We kept the joint model for the interactive demo because its contact benchmark shows better tight localization and fewer hidden reports. The two tests measure different things.
"""),
    ap_cell,
    md("""
## 4. Diagnose errors before changing the model

On frame 4021 a dancer supporting their weight on a hand and foot is spread across the floor. The tuned detector's visible body box is too short for one reference, and overlapping predictions can also go unmatched. Frame 4061 shows how several nearby bodies make the same problem worse. Amber is unmatched against these references. Some cases need human review before calling them background false positives.

This drove the visible-extent labeling rule. More labels improved development recall for hidden-contact dancers but did not carry over to the reserved diagnostic. The next controlled change is to correct a separate set of crowded frames, then test overlap handling at fixed settings.
"""),
    code(r"""
for key in ('published', 'tuned_300'):
    comparison = report['models'][key]['reserved_diagnostic']
    print(names[key], 'matched / reference boxes:', comparison['matched_at_025_iou50'], '/', comparison['reference_boxes'])
show(evidence / 'error_analysis/frame_4021.jpg')
"""),
    md("""
## 5. Project contacts onto a square stage

Four ordered coplanar correspondences give enough constraints for a perspective map. We project the support point through a 3×3 homography and divide by the last coordinate. The stage is 12 × 12 m, so both map axes use the same scale. The original calibration extrapolates the front corners. A jump breaks the floor-plane assumption, so the label has no measured point. The three new recordings register fixed venue landmarks to adjust the map during zooms and shifts.

[Original calibration](Experiment/Dance_CVAT_Pilot/calibration_draft.json) · [Camera registration](tools/register_external_stage.py)
"""),
    code(r"""
cal = load(exp / 'Dance_CVAT_Pilot/calibration_draft.json')
H = cal['pixel_to_relative_mat_homography']
def project_pixel(point):
    x, y, w = [sum(a*b for a, b in zip(row, [*point, 1])) for row in H]
    return round(12*x/w, 2), round(12*y/w, 2)
check = load(evidence / 'geometry/edge_check.json')
print('Corners on the map in meters:', [project_pixel(p) for p in cal['image_points']])
print(f"Median edge residual: {check['baseline_median_pixel_edge_error']:.2f} → "
      f"{check['draft_median_pixel_edge_error']:.2f} px on {check['independent_edge_picks_same_image']} separate edge picks.")
"""),
    md("""
## 6. Compare contacts and choose the replay threshold

The seven reserved images contain 46 visible and 50 hidden reference contacts. Missing visible predictions count as misses. The 300-image joint model improves the tight 1% result and makes fewer hidden reports. The published pipeline still leads at 3%. These references were checked on one performance and reused in prior comparisons.

The offline trajectory fit uses future frames to stabilize motion and bridge some gaps. A hollow dot means an estimated position. For the demo we lowered reportability to 0.30 after comparing the number of displayed dots and replay jitter. The table uses thresholds selected on validation.
"""),
    contact_cell,
    md("""
## 7. Fit formations and inspect the result

We started with horizontal rows and then allowed tilted lines. The detector searches for narrow groups of dots, fits a line to the current group and checks that it persists across nearby frames. The dashboard defaults to four dots and ±10°. A line's score drops with mean perpendicular offset. At 6% of stage width the score reaches zero. Hollow positions are excluded and the overall score is the mean across detected lines.

The score describes the geometry of the reported points. A choreography reference and independently labeled contacts would be needed to grade intended formations or timing.
"""),
    code(r"""
summary = load(evidence / 'replays/summary.json')
table(['New choreography', 'Frames processed', 'Frames with ≥1 contact', 'Frames with ≥1 line (4 dots, ±10°)'],
      [[r['title'], f"{r['frames']:,}", f"{r['frames_with_contacts']:,}", f"{r['frames_with_default_line']:,}"] for r in summary])
show(evidence / 'analytics/stage_occupancy.png')
"""),
    md("""
## 8. Run it live in the defense

Open the [dashboard](dashboard/index.html), switch to onare near 3:05, and change the angle or minimum dot count while paused. Compare it with Wicked Games during the dark opening. In the [upload app](README.md), process the bundled 12-second sample and open the downloaded contact CSV. Each row records frame, time, track ID, stage coordinates and contact state. The combined MP4 draws video and map on the same output frame.

Highest priority before submission: independent review of 30–45 frames and an annotated holdout from a different performance. See the [rubric checklist](docs/rubric_and_gaps.md), [casebook](docs/error_casebook.md), [decision log](docs/experiment_log.md), [AI disclosure](docs/genai_disclosure.md) and [defense notes](docs/defense_notes.md).
"""),
    code(r"""
for label, folder in [('RTX sample', 'rtx_sample'), ('CPU smoke test', 'cpu_smoke')]:
    meta = load(evidence / 'integration' / folder / 'metadata.json')
    print(f"{label}: {meta['frames']} frames, {meta['fps']:.0f} fps export; {meta['inference_fps']:.2f} fps inference")
display(Video(url='experiments/integration/rtx_sample/annotated_replay.mp4', width=930))
"""),
]

extra = [
    md("""
## 10. Evaluate boxes with standard AP

Compare COCO YOLO26m, the earlier dance detector and the 300-image fine-tune with the same tiles and image size. These are full-frame results after merging overlapping tiles. The training log reports validation on tiled images, which is a different protocol. Development has 197 reference boxes on 11 frames. The reserved diagnostic has 103 on seven frames. References are visually approximate and potentially incomplete. COCO detections also helped initialize some labels, which may favor the generic model.
"""),
    setup,
    ap_cell,
    code(r"""
show(evidence / 'detector/tuned_300/development/BoxPR_curve.png', 640)
show(evidence / 'detector_training/results.png', 830)
for split_name in ('development', 'reserved_diagnostic'):
    table(['Detector', 'Precision at 0.25', 'Recall at 0.25', 'Hidden-contact bodies matched'],
          [[names[key], f"{report['models'][key][split_name]['precision_at_025_iou50']:.3f}",
            f"{report['models'][key][split_name]['recall_at_025_iou50']:.3f}",
            f"{h['matched_boxes']}/{h['reference_boxes']}"]
           for key in names for h in [report['models'][key][split_name]['by_contact_at_025_iou50']['hidden']]])
"""),
    md("""
At confidence 0.25 the tuned detector matched more development bodies with hidden floor contact, from 38 to 51 out of 68. On the reserved diagnostic it fell from 34 to 30 out of 50, and generated more unmatched boxes. Frame 4021 shows the same issue around floor work and dense overlap. The [casebook](docs/error_casebook.md) records what changed, what did not improve, and which error needs independent adjudication.
"""),
    code(r"""
show(evidence / 'error_analysis/frame_4021.jpg')
show(evidence / 'error_analysis/frame_4061.jpg')
"""),
    md("""
## 11. Audit the labels, repeats and modeling choices

The final split has a temporal guard and exact image checks. The stage-crop similarity check flags visually close frames without declaring them independent. The learning-curve evidence spans the 60-, 200- and 300-image runs, which also changed the labels and training recipe. Treat this as an incremental comparison. A controlled dataset-size curve, detector freeze/LR sweep and matched augmentation ablation are listed as open in the [experiment matrix](docs/experiment_log.md).
"""),
    code(r"""
audit = load(evidence / 'data/split_audit.json')
for key, rows in audit['nearest_luma_signature'].items():
    closest = min(rows, key=lambda r: r['mean_absolute_luma_difference'])
    print(key, 'closest appearance pair:', closest)
repeat = load(early / 'label_repeatability/results.json')
print(f"Assistant repeat review: {repeat['same_contact_status']}/{repeat['samples']} states match; "
      f"median difference {repeat['median_pixel_difference']:.2f} px across {repeat['both_visible']} paired points.")
"""),
    contact_cell,
    md("""
## 12. Tracking, calibration and spatial analytics

The main demo uses the established identity and offline smoothing work. ByteTrack runs on new uploads because it starts from detected boxes without a manually seeded roster. BoT-SORT had fewer within-track gaps on the same 300-frame sample. The difference is about continuity; the identity diagnostic covers only four held-pose frames. A dedicated identity sequence and IDF1/HOTA remain open. Venue registration established usable camera alignment for the three later recordings. Training a landmark model remains a separate study.

The two occupancy maps use the same scale and a 0.6 m Gaussian bandwidth. Each covered frame contributes equal mass, with reflection at the boundary. Reported and estimated locations are plotted separately. Differences between the maps reveal where imputation changes the picture. Read the [extension evidence and limits](docs/rubric_and_gaps.md) before drawing conclusions about the choreography.
"""),
    code(r"""
diagnostic = load(early / 'diagnostics/latest_identity_diagnostic.json')
table(['Identity diagnostic', 'Matched boxes', 'Correct seed ID', 'Wrong seed ID'],
      [[name, row['matched_observed_boxes'], row['matches_with_correct_initial_id'], row['wrong_initial_id_matches']]
       for name, row in diagnostic['candidates'].items()])
tracking = load(evidence / 'tracking/comparison.json')['results']
table(['Same detections, 300 frames', 'Track IDs', 'Median frames per ID', 'Within-ID gaps'],
      [[name, row['track_ids'], row['median_frames_per_id'], row['within_id_gaps']]
       for name, row in tracking.items()])
smoothing = load(evidence / 'tracking/smoothing.json')['results']
table(['Same support inputs', 'Within 3%', 'P99 step change (% width)', 'Synthetic half-response lag'],
      [[name, f"{row['within_3pct']}/{row['visible_refs']}",
        f"{row['continuity_on_common_samples']['acceleration_p99']:.3f}",
        f"{row['synthetic_3pct_step_half_response_frames']} frames"]
       for name, row in smoothing.items()])
analytics = load(evidence / 'analytics/summary.json')
for status, values in analytics['statistics'].items():
    print(f"{status}: coverage {values['covered_frames']}/{values['all_frames']} frames")
show(evidence / 'analytics/stage_occupancy.png')
"""),
    md("""
The causal Kalman run localized 33/46 within 3% on the reused reserved images and reached half of the synthetic step one frame later. The offline fit localized 29/46 and produced the smoothest common samples. We kept the offline replay fit for visual review. The small diagnostic motivates an independent filter comparison before choosing a method for real-time feedback.
"""),
]

labeling = [
    md("""
# Label dancer boxes and floor contacts

Open one of the three 100-frame review packs in the [Windows annotator](annotator/README.md) or CVAT. Use the images folder and matching XML. See the [data card](docs/dataset_card.md) for scope and [the review protocol](docs/annotation_policy.md) for the independent agreement exercise.
"""),
    setup,
    md("""
## Review each visible body

Draw the visible extent of one distinguishable dancer, including visible hands or feet that extend the box. At an image boundary, clip to the image. In overlap, use a box only if you can distinguish the body. Leave a blurry or ambiguous support uncertain. Only visible floor contacts receive a point. Use the midpoint for two clear supports, and choose feet, knees, mixed knee and foot, palms, or seat as applicable.
"""),
    code("show(early / 'figures/label_examples.jpg')"),
    md("""
## Keep the ID and split

Match the point's positive image-local ID to its box. Mark hidden, airborne and uncertain on the box and leave the point absent. Floor guides describe the stage, so leave them with the frame. Export **CVAT for images 1.1**. Preserve the original exports and record reviewer, date, changed labels and open cases. Use the supplied script to recreate any pack.
"""),
    code(r"""
qa = load(trial / 'structural_qa.json')
for label, part in qa['parts'].items():
    print(f"Pack {label}: {part['frames']} images, {part['dancer_boxes']} boxes, {part['support_points']} points")
print('Build review ZIPs: python tools/build_300_packs.py')
"""),
]


def main():
    research_path = ROOT / "Research_Notebook.ipynb"
    research = nb.read(research_path, as_version=4)
    prior = research.cells[:next((i for i, cell in enumerate(research.cells)
                                  if cell.source.startswith(SECTION)), len(research.cells))]
    research.cells = prior + extra
    for filename, notebook in (("Master_Notebook.ipynb", nb.v4.new_notebook(cells=master, metadata=research.metadata)),
                               ("Labeling_Guide.ipynb", nb.v4.new_notebook(cells=labeling, metadata=research.metadata)),
                               ("Research_Notebook.ipynb", research)):
        nb.validate(notebook)
        nb.write(notebook, ROOT / filename)


if __name__ == "__main__":
    main()
