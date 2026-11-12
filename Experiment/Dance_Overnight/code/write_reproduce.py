from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
text='''DANCE STAGE MAPPING — REPRODUCTION

Start by opening START_HERE.html. The executed notebook reads saved evidence;
opening it does not launch training. Videos and CVAT review exports are local.

Keep the supplied folder structure. Code resolves paths relative to its files.
Use Python3.12. Install requirements.txt in a virtual environment. For GPU work,
install the matching CUDA PyTorch build from https://pytorch.org/get-started/locally/.
The recorded environment used torch2.11.0+cu128 on an RTX4070 Laptop GPU.
environment.json records the actual installed versions. Internet is needed for
pretrained ImageNet initialization when it is not already cached.

Run these commands from the Dance_Overnight folder:

1. Reproduce the final reserved scores from cached detector/support predictions:
   python code/reproduce_final.py --output reproduced_final
   This creates a new directory and checks exact agreement with saved scores.

2. Rerun the FIRST frozen support network on its saved detector/tracker boxes:
   python code/reproduce_selected.py --mode offline --device cpu --output fresh_support
   Use --device cuda for the original GPU. This performs fresh support inference,
   but does not claim to rerun YOLO or identity association.

3. Retrain the selected support architecture with the supplied56-frame manifest:
   python code/retrain_selected.py --name replication_support_s17 --seed17
   Correct syntax for the final option is: --seed 17
   The new run is separate. Checkpoints are selected on development validation.
   Do not replace frozen results with a newly trained model's outputs.

4. Rebuild report figures and the executed notebook after all saved results exist:
   python code/report_figures.py
   python code/build_report.py

5. Validate a human-reviewed CVAT export:
   python code/import_cvat_review.py annotations.xml reviewed_labels.json
   This saves a new annotation revision. Review it before training.

The full_replay/source.avi is the97-second lossless extracted static-camera section.
The long_sequence/source.avi is the shorter first reserved evaluation context.
Detector, support and tracking caches preserve the exact experiment inputs.
The original video is not required for these saved replays.

The full_replay.py script records the longer stress-test pipeline. It reuses
existing caches when present; invoking it is not a claim of completely fresh
end-to-end inference. Earlier experiments' selected configurations, training
curves and rejected candidates are saved for audit. Unselected model weights
may be omitted from the sharing package; retrain those runs to recreate them.

Evaluation limits: one video, approximate assistant labels, correlated frames,
incomplete crowd annotation, relative mat dimensions, and offline future-frame
filtering. Hidden/airborne contacts have no manual floor coordinate. Formation
recognition and per-dancer formation accuracy remain future work.
'''
text=text.replace(' --seed17\n   Correct syntax for the final option is: --seed 17',' --seed 17').replace('Python3.12','Python 3.12').replace('torch2.11','torch 2.11').replace('RTX4070','RTX 4070').replace('supplied56','supplied 56').replace('the97','the 97')
latest='''LATEST CONFIRMATION AND TEAM REVIEW

The newest frozen comparison is final_confirmation/results.json.
Its exact selected detector, support head, thresholds and filter are recorded in
final_confirmation/frozen_selection.json. The previous frozen results stay intact.

Reproduce the newest map fit and confirmation scores from the supplied caches:
  python code/reproduce_latest.py --output reproduced_latest

Retrain the expanded 60-frame support architecture under a new run name:
  python code/retrain_selected.py --variant expanded --name replication_expanded_s17 --seed 17
This uses the original experiment manifest, not later human-review revisions.
The report distinguishes this model from the established 56-frame model below.

The master CVAT review contains 60 training images and 11 validation images.
Use the separate teammate_A, teammate_B and teammate_C packs for parallel work:
20 training images each, plus 4 / 4 / 3 validation images. Keep their split tags.
After exporting each task, validate and merge with:
  python code/import_cvat_review.py A.xml A.json
  python code/import_cvat_review.py B.xml B.json
  python code/import_cvat_review.py C.xml C.json
  python code/merge_review_exports.py combined_review.json A.json B.json C.json
Prepare a new training revision after the actual human review:
  python code/prepare_reviewed_training.py combined_review.json data/team_review_v1.json --reviewer "Names and review date"
Then, from the FULL experiment package with GPU dependencies installed:
  python code/retrain_selected.py --variant expanded --manifest data/team_review_v1.json --name team_review_v1_s17 --seed 17
This creates new weights and preserves all frozen experiment outputs. Validate
the new run before replacing the demonstration model. The small labeling ZIP
does not include the GPU training environment or model code.
The supplemental_ending export is an older
separate copy of six images already included in the main task; do not import twice.
Three reserved evaluation exports remain outside training.

The full late replay is videos/late_full_section.mp4. The short fresh point
comparison is videos/final_confirmation_point_ablation.mp4. Solid map dots are
model outputs, hollow dots are estimates, and unavailable locations are omitted.

Temporal-context experiments, if present, are separate development experiments.
They do not silently replace the frozen confirmation model. Their matched
repeated-current-frame control is included in temporal_experiment/results.json.

EARLIER FROZEN EXPERIMENTS

'''
text=text.replace('Run these commands from the Dance_Overnight folder:\n\n','Run these commands from the Dance_Overnight folder:\n\n'+latest).replace('and11','and 11').replace('are20','are 20')
(OUT/'REPRODUCE.txt').write_text(text,encoding='utf-8')
(OUT/'requirements.txt').write_text('numpy==2.5.2\nscipy==1.18.1\nopencv-python==5.0.0.93\ntorch==2.11.0\ntorchvision==0.26.0\nultralytics==8.4.178\nPillow==12.3.0\nmatplotlib==3.11.2\npandas==3.0.6\nnbformat==5.11.1\nnbclient==0.11.0\nipykernel==7.4.0\nimageio-ffmpeg==0.6.0\nrequests\n',encoding='utf-8')
print('Reproduction instructions written')
