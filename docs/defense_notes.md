# Defense walkthrough

Start in the [master notebook](../Master_Notebook.ipynb). Show the live app on the bundled clip and the saved dashboard on onare near 3:05. Change the angle and dot count while paused, then show the dark opening of Wicked Games. Open one case with missed or unmatched boxes and trace the cause to the labeling rule and model comparison.

| If asked | Code to open | What to explain |
|---|---|---|
| Why tile the image? | [detector_probe.py](../Experiment/Dance_Overnight/code/detector_probe.py) | Crop scale helps small people. Offsets restore full-frame coordinates. Duplicate suppression can also remove genuine overlap. |
| What was fine-tuned and how was it chosen? | [train_300.py](../Experiment/lighting_expansion_300/train_300.py), [AP script](../tools/evaluate_submission_detector.py) | Backbone features, box/class outputs, frozen layers, validation, IoU, PR curve and the AP50 versus AP50:95 tradeoff. |
| Why a support point and a visibility score? | [candidate_evaluate.py](../Experiment/Dance_Overnight/code/candidate_evaluate.py) | The crop head predicts heatmap coordinates and reportability. A knee, seat or palm may carry weight. Explain when to abstain. |
| Why four corners? | [projection pipeline](../src/pipeline.py) | Ordered non-collinear points constrain an eight-degree-of-freedom plane transform. Divide by the homogeneous scale. Show why a raised point cannot be projected as a floor contact. |
| How are dots and lines synchronized? | [dashboard engine](../dashboard/formation.js), [fresh pipeline](../src/pipeline.py) | One source frame index per output frame; line membership uses solid points; the offline fit can use future frames. |
| What's the strongest limitation? | [rubric and gaps](rubric_and_gaps.md) | Same-performance approximate labels and a small reserved set. New-video replays show generalization qualitatively. Identity and scoring need separate references. |

Suggested rehearsal split: one person opens data and detector code, one opens contact and geometry code, and one drives the apps and failure case. Rotate so every teammate runs a clip and answers a question without AI assistance. Record actual owners, reviewers, dates and any corrections after rehearsal.
