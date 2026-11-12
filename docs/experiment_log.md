# Experiment and decision log

| Question | What we observed | Decision and evidence |
|---|---|---|
| Can we use the bottom of the dancer box? | Geometry alone missed floor-work support and reported more hidden contacts. | Keep boxes for detection and learn contact position plus reportability. [Support ablations](../Research_Notebook.ipynb) |
| Does appearance and motion help tracking? | The expanded detector increased initial-ID matches in the short development check. Held-pose ID checks leave crowded crossings unresolved. | Keep original replay for that roster; use automatic ByteTrack for new rosters. [Held-pose results](../Experiment/Dance_Overnight/diagnostics/latest_identity_diagnostic.json) |
| Do more frames or models help the support head? | Neighbor-frame inputs and the tested seed ensemble did not beat the chosen head on their development protocols. Crop averaging helped the chosen model and increases compute. | Keep the single-frame checkpoint with crop averaging. [Temporal](../Experiment/Dance_Overnight/temporal_experiment/results.json) · [Seeds](../Experiment/Dance_Overnight/seed_repeatability/results.json) · [Efficiency](../Experiment/Dance_Overnight/efficiency/results.json) |
| How should we handle visibly clipped or absent supports? | The user noticed bad box bottoms and omitted floor guides. Visual review corrected labels and the floor guides were kept in the review export. | Use visible-extent boxes and explicit visibility states. [Revision evidence](../Experiment/lighting_expansion_200/visual_relabel/visual_check.json) · [Merged audit](../Experiment/lighting_expansion_300/structural_qa.json) |
| What changed after 200 and 300 frames? | The 300-image joint run improved the tight 1% contact result and reduced hidden reports. The published baseline led at 3%. The generic detector led the stricter AP average. | Use the joint demo and retain both comparisons. [Contact results](../Experiment/lighting_expansion_300/RESULTS.md) · [Full-frame AP](../experiments/detector/comparison.json) |
| Can more dots be shown without distracting flicker? | The 200-frame threshold sweep showed more coverage at 0.30; movement diagnostics changed modestly. | Use 0.30 in the replay. Benchmark thresholds are selected separately on validation. [Sweep](../Experiment/lighting_expansion_200/runs/lighting_200_visual_s17/display_sweep.json) |
| How should rows adapt to choreography? | Fully horizontal lines missed intentional tilt. A 30° limit also admitted some less useful fits in review. The user confirmed a square stage. | Refit under an adjustable angle with ±10° and four points as the defaults, using the 12 × 12 m aspect. [Dashboard](../dashboard/index.html) |
| Will it run on another MP4? | The three additional videos processed end to end with the same weights and per-frame venue registration. Wicked Games has fewer reported dots during its dark opening. | Include all replays and keep new-performance evidence qualitative. [Replay evidence](../experiments/replays/summary.json) |

The early pilot, 200-image revision and 300-image walkthrough are preserved in the project snapshot and historical experiment paths. [Training logs](../experiments/training_logs) · [Model settings](../experiments/detector_training/args.yaml)

| Roadmap experiment | Status |
|---|---|
| E0 generic detector and E1 tuned models | Same-tile AP and PR curves for generic, earlier dance and 300-frame models. Earlier dance training involved several stages. |
| E2 more training frames | 60/200/300 history and comparison. Counts, labels and recipes changed together, so a fixed-recipe size curve remains open. |
| E3 freeze and learning rate | Final model has recorded settings. A matched new sweep remains open. |
| E4 augmentations | Earlier lighting trials and later augmentation settings are preserved. A matched detector blur/brightness/scale ablation remains open. |
| E5 support estimates | Box geometry, pose-related pilot code, learned contact, visibility, and perturbation evidence. |
| E6 tracking and repeats | Same-detections tracker comparison, held-pose ID audit, robust smoothing and seed/neighbor experiments. |
