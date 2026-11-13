# Rubric and submission status

Formation Review helps dance judges assess group formation accuracy by measuring alignment in detected rows. Floor contacts provide the stage positions used to fit each row. The live heat map adds a view of stage use over time. The team roadmap records verbal approval to use dance for the challenge. [Class brief](briefs/CV_Challenge_2026_2027.pdf) · [Team roadmap](briefs/Dance_CV_Challenge_Team_Roadmap.pdf)

| Rubric | Evidence ready to defend | Still open |
|---|---|---|
| Data and annotation · 25% | 60 train + 11 validation pilot, 200-frame revision, 300-frame merged export, A/B/C packs, split audit, and a 36-frame AI review of selected dancers. [Audit](../experiments/final_data_review/REPORT.md) | Two human reviewers can use the blank pack to measure independent agreement and adjudicate difficult overlaps. |
| Modeling · 35% | Three YOLO checkpoints compared at AP50 and AP50:95, training curves, contact and temporal ablations, and a frozen ten-frame Wicked Games evaluation with approximate AI references. [Results](../experiments/final_evaluation/results.json) | Independent human references and more performances would strengthen the external comparison. A fixed-recipe dataset-size curve and isolated freeze, learning rate and augmentation sweeps remain open. |
| Failure analysis · 20% | Detector cases, visibility strata, lighting examples, and a matched merge-IoU comparison on the new video. Reducing merge IoU to 0.40 removed one unmatched prediction and lost one matched dancer. [Results](../experiments/final_evaluation/results.json) | Human adjudication can establish which unmatched predictions correspond to real bodies and whether the contact errors recur. |
| Pipeline and homography · 10% | Four-point projection, 14 original and 38 replay edge picks, synchronized uploads and exports, CPU and RTX evidence, and seven known formation cases. The extracted dashboard was rehearsed on BESTIJA. [Formation checks](../experiments/final_formation_validation/README.md) · [Product QA](../experiments/final_product_validation/README.md) | A rehearsal on another team's computer and measured stage landmarks would extend validation. |
| Process and ownership · 10% | Decision log, version history, reproducible commands, AI disclosure, role-based rehearsal and individual prompts. [Handoff](final_handoff.md) | Teammates confirm their own contributions and complete the course declaration and live defense requirements. |

## Extensions

- **Tracking and trajectories.** ByteTrack runs in the upload app. A same-detections ByteTrack/BoT-SORT continuity experiment and the earlier held-pose ID diagnostic are included. Add continuous human identity references before reporting ID switches, IDF1 or HOTA.
- **Automated keypoints.** Venue landmark registration adjusts camera framing on three new clips. The [alignment log](../experiments/replays/calibration.log) and [edge check](../experiments/geometry/edge_check.json) show what was checked. Training a stage landmark model requires independently labeled, repeatedly visible floor anchors. The extrapolated front corners and limited venue coverage make that a separate study.
- **Tactical analytics.** Judges can review row alignment, then use the live heat map and detected coverage to see how widely the group used the stage. The research notebook compares occupancy for reported and estimated positions. Symmetry, timing and alignment against a planned choreography would require separate references.

## Independent checks for the team

1. Independently label the included 36-frame sample and adjudicate. Review the ten Wicked Games references for the external comparison.
2. Rehearse a fresh upload, one detector failure, the geometry of a fitted row and the stage-use heat map. Record who can explain and reproduce each part.
3. Verify each personal contribution, footage sharing permissions and the applicable course declaration.
