# Rubric and submission status

Requirements are from the [class brief](briefs/CV_Challenge_2026_2027.pdf) and [team roadmap](briefs/Dance_CV_Challenge_Team_Roadmap.pdf). The roadmap records verbal approval to use dance. Our scope is reviewing floor positions, spacing and fitted formations on a 12 × 12 m stage.

| Rubric | Evidence ready to defend | Still open |
|---|---|---|
| Data and annotation · 25% | 60 train + 11 validation pilot, 200-frame revision, 300-frame merged XML and images, A/B/C packs, visual examples, split and file hash audit | Independent team review and 30–45 double-labeled frames, including adjudication of difficult overlaps. A separate performance with labeled test frames |
| Modeling · 35% | Generic, earlier dance and 300-frame YOLO compared with AP50 and AP50:95 under the same full-frame settings; training curves; contact and temporal ablations | Freeze, learning rate and augmentation sweeps with one factor changed at a time; true dataset-size learning curve and independently reviewed test |
| Failure analysis · 20% | Side-by-side detector cases, per-contact-stratum box recall, hidden-report counts, lighting examples and documented unsuccessful approaches | Adjudicate unmatched predictions that could be missing labels; verify whether the suggested overlap changes improve independent examples |
| Pipeline and homography · 10% | Four-point projection, 14 separate edge picks, synchronized Streamlit upload output, frame/CSV/calibration exports, CPU and RTX run evidence, four-replay dashboard | Rehearse on a teammate's clean machine; gather surveyed or independently identifiable stage landmarks if meter-level accuracy matters |
| Process and ownership · 10% | Decision log, version history, reproducible commands, AI disclosure and individual walkthrough prompts | Confirm contributions and exact university AI disclosure format; all three members perform a live walkthrough and record feedback |

## Extensions

- **Tracking and trajectories.** ByteTrack runs in the upload app. A same-detections ByteTrack/BoT-SORT continuity experiment and the earlier held-pose ID diagnostic are included. Add continuous human identity references before reporting ID switches, IDF1 or HOTA.
- **Automated keypoints.** Venue landmark registration adjusts camera framing on three new clips. The [alignment log](../experiments/replays/calibration.log) and [edge check](../experiments/geometry/edge_check.json) show what was checked. Training a stage landmark model requires independently labeled, repeatedly visible floor anchors. The extrapolated front corners and limited venue coverage make that a separate study.
- **Tactical analytics.** The dashboard fits and scores rows and the research notebook compares KDE occupancy for reported versus estimated points. Spacing is descriptive. Symmetry and transitions against choreography references remain open. Pose and timing synchronization would require a separate human-annotated task.

## Finish in this order

1. Independently label a shared 30–45 frame sample from the 300 and adjudicate. Use a separate set of fully annotated frames from another performance as a locked test. Give extra attention to dense overlaps and the darker opening of Wicked Games.
2. Rerun the three detector checkpoints on that locked split. Train a controlled ablation only if it answers a specific question, such as whether overlap-aware suppression improves precision without losing tightly grouped dancers. Keep confidence and reference protocol fixed for comparison.
3. Rehearse the uploaded MP4, one detector mistake and the homography explanation on a teammate's machine. Record who can explain and reproduce each part, verify footage sharing rights, and complete the course AI declaration.
