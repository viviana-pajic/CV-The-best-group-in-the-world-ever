"""Refresh the defense walkthrough and append the submission evidence to research."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
SECTION = "## 10. Evaluate boxes with standard AP"
RESEARCH_INTRO = """
# Research behind Formation Review

Formation Review helps dance judges assess group formation accuracy. The experiments below test how reliably we can locate dancers on the stage and how those positions support alignment scores and the live stage-use heat map. The contact benchmarks measure the quality of the position estimates that feed the formation analysis.

The cells read saved results. The [master notebook](Master_Notebook.ipynb) walks through the judging tool, and the [300-image results](Experiment/lighting_expansion_300/RESULTS.md) give the training details.
""".strip()


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
# Formation Review for dance judges

Formation Review helps dance judges assess group formation accuracy. Judges can pause the video, see fitted lines on the 12 × 12 m stage map, and review how closely the dancers align in each detected row.

Floor contacts give us the stage positions used to fit those lines. The live heat map adds a view of how widely the group uses the stage and where dancers spend the most time.

We developed the tool on the original fixed-camera choreography and visually checked it on Wicked Games, onare and Dernek. The team roadmap records verbal approval to use dance for the challenge.

[Interactive replay](dashboard/index.html) · [Research evidence](Research_Notebook.ipynb) · [Labeling guide](Labeling_Guide.ipynb) · [Submission handoff](docs/final_handoff.md)

The cells below read saved experiment evidence and run on CPU. [Run a fresh clip](README.md) to execute the detector, tracker, support model, map and export from an MP4.
"""),
    setup,
    md("""
## 1. Locate the dancers on the stage

Generic YOLO finds many standing dancers. Overlapping crops help keep small performers large enough for inference, then cross-crop IoU suppression merges duplicates. Tracking links detections across frames. We first used the bottom of each box as the dancer's floor position. During floor work, a knee, seat or palm may be the clearest point to project onto the stage. The learned head uses a body crop to estimate that point and whether the frame provides enough evidence to report it.

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

The final audit reviewed one selected dancer in each of 36 raw frames, 12 from each batch. The AI reviewer recorded their judgments before opening the current labels. The review pack is ready for independent team annotation. [Audit report](experiments/final_data_review/REPORT.md) · [Blank review pack](data/final_annotation_review/README.md)
"""),
    code(r"""
audit = load(trial / 'structural_qa.json')
split = load(evidence / 'data/split_audit.json')
table(['Pack', 'Frames', 'Boxes', 'Visible contact points'],
      [[name, row['frames'], row['dancer_boxes'], row['support_points']] for name, row in audit['parts'].items()])
print(f"{split['unique_verified_training_files']} verified unique training files; "
      f"{split['minimum_train_validation_gap_frames']} frames at the closest train/validation boundary.")
print(f"Reserved exclusion: {split['reserved_excluded_inclusive']}. Same performance across splits.")
review = load(evidence / 'final_data_review/comparison.json')['overall']
print(f"Targeted AI review: {review['contact_state_agreement']}/{review['contact_state_denominator']} matched contact states. "
      f"Median point difference was {review['point_distance_median_px']:.1f} px across {review['joint_visible_points']} visible pairs.")
"""),
    md("""
## 3. Fine-tune and compare the box detectors

The 300-frame detector starts from the earlier dance checkpoint and fine-tunes on tiled dancer boxes. It freezes the first ten layers, uses AdamW with learning rate 0.00025, and keeps mosaic off. For the comparison below we reran all three checkpoints with the same tiles, 960-pixel inference and merge IoU. AP integrates the precision–recall curve at an IoU threshold. AP50:95 averages ten IoU thresholds, so it penalizes loose boxes more.

The 300-frame model leads development AP50. Generic YOLO leads AP50:95 on the two original diagnostic splits. The joint model used in the interactive demo had better tight localization and fewer hidden reports on the original contact comparison.

The later Wicked Games sample contains ten frames held out from training and reviewed visually by an AI contributor before scoring. Generic YOLO reached 0.965 AP50 and 0.622 AP50:95. The 300-frame detector reached 0.932 and 0.492 on the same references. [Wicked Games results](experiments/final_evaluation/results.json)
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

The Wicked Games check exposed a transfer weakness. A box-bottom baseline landed within 0.48 m of 20 of 29 visual references. Solid replay dots reached 7 of 29, and the replay reported fewer positions when the reference contact was uncertain. That trade-off matters when a judge reviews a row under difficult lighting. [Wicked Games results](experiments/final_evaluation/results.json)
"""),
    contact_cell,
    md("""
## 7. Measure formation alignment

We started with horizontal rows and then allowed tilted lines. The detector searches for narrow groups of dancer positions, fits a line to each group and checks that it persists across nearby frames. The dashboard defaults to four dancers and ±10°. A row scores 100% when every reported position lies on the fitted line. The score falls as the mean perpendicular distance grows and reaches zero at 6% of stage width. The overall score averages the detected rows.

Judges can use these scores to assess how cleanly dancers form rows, taking the intended choreography into account. The additional heat map shows cumulative stage use up to the playhead, with warmer areas showing more time spent there.

Seven known formations verified the scoring engine. In a four-person row, moving one dancer 0.48 m off the line lowered the score to 83.33%. Moving the whole straight row by 0.48 m kept its alignment at 100%. The stage map lets a judge inspect the row's placement. [Formation validation](experiments/final_formation_validation/README.md)
"""),
    code(r"""
summary = load(evidence / 'replays/summary.json')
table(['New choreography', 'Frames processed', 'Frames with ≥1 contact', 'Frames with ≥1 line (4 dots, ±10°)'],
      [[r['title'], f"{r['frames']:,}", f"{r['frames_with_contacts']:,}", f"{r['frames_with_default_line']:,}"] for r in summary])
example = load(evidence / 'final_formation_validation/onare_comparison.json')
reference = example['same_dancers_visual_reference'][0]['score']
print(f"onare at {example['time_s']:.2f} s · Six dancers scored {example['dashboard_line_score_pct']:.2f}% "
      f"from detected positions and {reference:.2f}% from approximate visual picks.")
show(evidence / 'analytics/stage_occupancy.png')
"""),
    md("""
## 8. Run it live in the defense

Open the dashboard and pause onare near 3:05 to inspect the fitted row and its alignment score. Change the angle or minimum dancer count and see how the fit changes. Then scroll to the live heat map and scrub through Dernek to see stage coverage build. Use the bundled 12-second clip in the upload app to demonstrate how an MP4 becomes a synchronized replay and position export. [Dashboard](dashboard/index.html) · [Upload app](README.md)

The submission handoff links to the AI-assisted label audit, Wicked Games evaluation, and product verification. The team can use the included review pack for independent human agreement and complete the personal contribution records. [Submission handoff](docs/final_handoff.md) · [Rubric checklist](docs/rubric_and_gaps.md) · [AI disclosure](docs/genai_disclosure.md)
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
    md("""
## 13. Check the judging measurements

Known stage positions test the line fitting and scoring independently of the detector. The onare example compares the production score at one previously viewed frame with approximate visual references. The edge check compares the original static calibration with the registered camera views. Visible edge residuals are approximate pixel measurements.
"""),
    code(r"""
folder = evidence / 'final_formation_validation'
formation = load(folder / 'summary.json')
table(['Known stage positions', 'Expected rows', 'Fitted rows', 'Alignment scores'],
      [[r['case'], r['expected_rows'], r['fitted_rows'], ', '.join(f'{v:.2f}%' for v in r['scores_pct']) or '—']
       for r in formation['cases']])
registration = load(folder / 'registration_check.json')['cases']
table(['Replay', 'Edge picks', 'Static median error (px)', 'Registered median error (px)'],
      [[r['clip'], r['points'], r['static_median_px'], r['registered_median_px']] for r in registration])
"""),
    md("""
The registered calibration lowered the median error in the three sampled frames. The left edge of onare still had a 17.16 px residual, so treat the stage positions as estimates. The synthetic translated row retains a perfect alignment score because each dancer stays on the fitted row. Measuring placement against choreography needs a planned reference row. [Validation evidence](experiments/final_formation_validation/README.md)
"""),
    md("""
## 14. Evaluate a performance outside training

Ten Wicked Games frames were selected by time. An AI contributor reviewed the clean frames and froze the box and contact references before scoring predictions. The reference set covers a dark opening, a title overlay and later choreography. This video was used for earlier demonstrations, and the visual references are approximate. [Protocol](experiments/final_evaluation/protocol.json)
"""),
    code(r"""
external = load(evidence / 'final_evaluation/results.json')
table(['Detector', 'AP50', 'AP50:95', 'Unmatched predictions at .25', 'Missed boxes at .25'],
      [[names[key], f"{r['ap50']:.3f}", f"{r['ap50_95']:.3f}", r['false_positives'], r['false_negatives']]
       for key in names for r in [external['detectors'][key]['all']]])
contacts = external['contacts']
table(['Position method', 'Within 0.48 m', 'Median error on paired positions', 'Reported when reference uncertain'],
      [[label, f"{r['within_threshold_m']['0.48']}/{contacts['reference_points']}", f"{r['paired_median_error_m']:.3f} m",
        f"{r['reported_when_reference_uncertain']}/{contacts['uncertain_reference_points']}"]
       for key, label in [('box_bottom', 'Box bottom'), ('dashboard_solid', 'Solid dashboard contacts')]
       for r in [contacts['methods'][key]]])
"""),
    md("""
Lowering the tuned detector's cross-tile merge IoU from 0.55 to 0.40 removed one unmatched box and lost one matched dancer. AP50 fell from 0.932 to 0.900. The replay still uses 0.55. The contact results expose a trade-off between coverage, position error and reporting uncertain contacts. A broader human-reviewed set is needed before using these comparisons to change the judging pipeline. [Results and cases](experiments/final_evaluation/results.json)
"""),
    code(r"""
raw = contacts['post_hoc_raw_head']
table(['Post-hoc check on the same 20 visible cases', 'Median position error'],
      [[label, f"{raw['paired_medians_m'][key]:.3f} m"]
       for key, label in [('box_bottom', 'Box bottom'), ('raw_head', 'Raw contact head'), ('dashboard_solid', 'Solid replay dot')]])
"""),
    md("""
The five largest paired replay errors were already large in the raw contact estimates. This points to a contact-placement problem before smoothing on those cases. [Raw-cache diagnostic](experiments/final_evaluation/results.json)

## 15. Audit a wider sample of training labels

The AI reviewer looked at 36 raw frames, selected one dancer per frame, and recorded estimates before comparing with the merged annotation file. Matching allowed one unambiguous existing box per target. This measures disagreement on a targeted sample. The blank pack supports a separate full human review.
"""),
    code(r"""
review = load(evidence / 'final_data_review/comparison.json')['overall']
print(f"{review['frames_reviewed']} frames · {review['blind_box_count']} estimated boxes · "
      f"{review['matching_counts']['matched']} unambiguous matches")
table(['Agreement measure', 'Result'], [
    ['Median matched box IoU', f"{review['box_iou_median']:.3f}"],
    ['Contact-state agreement', f"{review['contact_state_agreement']}/{review['contact_state_denominator']}"],
    ['Median difference on jointly visible points', f"{review['point_distance_median_px']:.1f} px"],
    ['90th-percentile difference on jointly visible points', f"{review['point_distance_p90_px']:.2f} px"],
])
"""),
    md("""
The longer tail of point disagreements gives the team concrete cases to inspect. [Audit and discrepancies](experiments/final_data_review/REPORT.md) · [Blank review pack](data/final_annotation_review/README.md)
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
    prior[0].source = RESEARCH_INTRO
    research.cells = prior + extra
    for filename, notebook in (("Master_Notebook.ipynb", nb.v4.new_notebook(cells=master, metadata=research.metadata)),
                               ("Labeling_Guide.ipynb", nb.v4.new_notebook(cells=labeling, metadata=research.metadata)),
                               ("Research_Notebook.ipynb", research)):
        nb.validate(notebook)
        nb.write(notebook, ROOT / filename)


if __name__ == "__main__":
    main()
