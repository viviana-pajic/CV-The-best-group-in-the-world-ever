# Wicked Games new-performance diagnostic

Formation Review depends on finding dancers and placing their positions on the stage. We sampled ten frames across Wicked Games, a performance excluded from training and previously used for dashboard demonstrations. One AI contributor drew 54 visible-extent boxes and 29 discernible floor supports from clean frames before viewing or running predictions. A further 25 supports were unresolved because of light, occlusion or apparent jumps. The frame indices, policies, reference files and video were frozen by hash before evaluation. The references are approximate, especially for partly visible dancers, and still need human verification. The diagnostic covers one performance at the same venue. [Protocol](protocol.json) · [Reference annotations](annotations.json) · [Frozen hashes](frozen-inputs.json) · [Annotated review sheet](figures/reference_review.jpg)

## Detector comparison

All three checkpoints used the earlier AP evaluator's overlapping crops, 960-pixel model input, 0.06 candidate floor and 0.55 cross-crop merge IoU. The last three columns use confidence 0.25 and match IoU 0.50. The negative title frame remains in the evaluation.

| Detector | AP50 | AP50:95 | Matched | Unmatched predictions | Missed references |
|---|---:|---:|---:|---:|---:|
| Generic YOLO26m | 0.965 | 0.622 | 51 | 1 | 3 |
| Published dancer model | 0.936 | 0.495 | 49 | 4 | 5 |
| 300-frame dancer model | 0.932 | 0.492 | 49 | 9 | 5 |

Generic YOLO26m leads this sample on both AP measures. The 300-frame model's AP50:95 rises from 0.400 over the four opening frames to 0.549 over the six later frames. Five of its nine unmatched boxes occur on frame 50, four directly over title graphics. The later error examples include a missed body partly visible through smoke and a close seated overlap. A tighter cross-crop merge at 0.40 on the same 300-frame candidate boxes removed one unmatched prediction and lost one matched dancer. AP50 fell to 0.900. The diagnostic supports retaining 0.55 for the current replay. [Detection cases](figures/detector_cases.jpg) · [Metrics and per-frame results](results.json)

The detection figure shows unmatched predictions in red and missed reference boxes in green. An unmatched prediction is an error relative to these approximate labels. Box boundary uncertainty can affect the stricter IoUs and the assignment in overlaps.

## Floor positions

The floor comparison uses the same observed track boxes matched to the references at IoU 0.50. The simple baseline projects each matched box's bottom center onto the stage. The dashboard measurement uses its deployed solid contacts at reportability 0.30 after trajectory fitting. Both use the same per-frame calibration and 12 m scale. Missing contacts count against the two distance thresholds.

| Method | Reported on 29 visible references | Within 0.24 m | Within 0.48 m | Median error on the 20 common references | Reported on 25 uncertain references |
|---|---:|---:|---:|---:|---:|
| Box-bottom center | 28 | 12 | 20 | 0.269 m | 21 |
| Dashboard solid contacts | 20 | 6 | 7 | 0.602 m | 9 |

The box-bottom baseline localizes more visible references on this sample. The dashboard emits fewer points where reference support is unresolved. The uncertain cases include both genuinely difficult contacts and likely jumps, so their reported-point count describes abstention behavior. The meter distances inherit the stage calibration and annotation uncertainty. This experiment evaluates the input positions for formation review. Judge-scored formation accuracy and timing would require separate references. [Per-reference measurements](results.json)

### Raw proposal check

We checked the saved raw-contact cache after scoring the frozen benchmark. Its source and checkpoint hashes match the deployed replay, and the matched observed boxes agree to rounding. On the 20 common references the median raw-head error was 0.614 m. The five largest paired dashboard errors were 1.77–2.73 m, and the corresponding raw errors were already 1.72–2.68 m. In four visible examples, raw and displayed points lie close together above the reviewed ground support. This suggests that point placement accounts for much of those misses. In the full set of 29 visible references, one lacked a matched observed box, five raw points were below 0.30, two passing raw points projected outside the stage, and one passing in-stage point was absent after trajectory fitting. Nine raw proposals were within 0.48 m, and seven solid replay points met that tolerance. [Contact cases](figures/contact_cases.jpg) · [Raw cache diagnostics](results.json)

The contact figure uses green for the reviewed reference, orange for the box-bottom point, cyan for the displayed solid point and a magenta ring for the raw proposal.

## Reproduce locally

Use the pinned ML environment from the project README. The raw cache is packaged at `experiments/replays/cache/CbpSMoPItu0.inference.json` and was generated with the saved dashboard replay. In the working checkout the same cache is available under `build/external-test-videos/`.

```powershell
python tools/evaluate_formation_holdout.py --raw-cache experiments/replays/cache/CbpSMoPItu0.inference.json
```

The command verifies all frozen input hashes, scores the three detector checkpoints and the suppression variant, and checks the existing replay against the raw cache. It uses CUDA device 0 by default. `--device cpu` selects CPU inference. `--output` writes the comparison to another JSON path. Omitting `--raw-cache` runs the predeclared comparison without the exploratory raw proposal check.
