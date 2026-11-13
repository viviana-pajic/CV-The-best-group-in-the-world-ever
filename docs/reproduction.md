# Local setup and reproduction

Use Python 3.12. The RTX runs used BESTIJA, NVIDIA GeForce RTX 5080 and PyTorch 2.11.0+cu128. The app has a CPU option. Full precision and mixed precision runs may produce small score differences. The notebook path needs only the smaller requirements file.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-app.txt
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

For a CPU setup, use the PyTorch CPU wheel index in the third command and select CPU. The app expects a constant-rate 16:9 MP4 with a visible stage. Choose the bundled 12-second sample for a first run. Use four visible corners in the required order for fixed-camera material on a square stage. Automatic venue registration uses the original video and calibration in this folder.

```powershell
python scripts/run_demo.py --output build/fresh_demo
python tools/run_notebooks.py
python tools/evaluate_submission_detector.py --output build/detector_recheck
python tools/build_300_packs.py --batch combined
python scripts/compare_trackers.py
python scripts/compare_smoothing.py
python scripts/build_research_evidence.py
python -m unittest discover -s tests -v
```

The [300-frame recipe](../Experiment/lighting_expansion_300/REPRODUCE.md) documents how to retrain. A historical contact refit uses `python Experiment/Dance_Overnight/code/reproduce_latest.py --output build/refit`; this reuses saved detection and support caches. `scripts/run_demo.py` loads both checkpoints and performs fresh inference. The [integration outputs](../experiments/integration) include GPU sample and CPU smoke test metadata, frame exports, MP4 and calibration.

The interactive HTML dashboard reads saved replays. Double-click **Open Formation Review.cmd** from the extracted package on Windows. From a terminal, run `python open_dashboard.py` in the package root. The local server supports seeking through MP4s. The Windows annotation app is portable. [Dashboard guide](../dashboard/README.md) · [Annotation app guide](../annotator/README.md)

[Model inventory](../models/checkpoints.json) · [Configuration snapshots](../configs) · [Project commit history](git-history.txt)

## Verification on BESTIJA

- All three notebooks executed from the assembled submission folder with saved outputs.
- Fresh inference processed the 300-frame sample on RTX and a 16-frame sample on CPU. The Streamlit test ran the bundled sample and exposed all five download buttons.
- Geometry, split, timing, export and labeler tests passed. Four dashboard clips passed frame and line checks. A synthetic empty stage exported an empty contact table and synchronized video. Wrong aspect ratio, corrupt input, variable frame rate, absent venue anchors and a missing checkpoint produced explicit errors.
- The historical confirmation was reproduced from frozen caches. For earlier material the package uses the preserved repository. Some separate OneDrive-only archives were unavailable on this machine. A same-input Kalman comparison was rerun from the accessible frozen evidence.

The final QA used an extracted local package and includes browser playback, live heat map, launcher, upload and export checks. For the new evidence, run `node scripts/validate_formation.cjs`, `python scripts/check_replay_calibration.py`, and `python scripts/audit_reference_formation.py`. The training audit and Wicked Games evaluation have their own protocols. [Product QA](../experiments/final_product_validation/README.md) · [Training audit](../experiments/final_data_review/REPORT.md) · [Wicked Games protocol](../experiments/final_evaluation/protocol.json)

Human annotation agreement, team-machine setup and individual defense rehearsal require the team.
