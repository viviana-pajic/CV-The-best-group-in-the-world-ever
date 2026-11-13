# Training annotation audit

An AI reviewer selected one dancer in each of 36 raw training images, recorded the visible-extent box and floor-contact judgment, then compared those observations with the existing merged export. This checks the stage positions that feed formation alignment. It covers one selected dancer per image. Independent human agreement and full-frame reannotation still require separate review.

The sample contains 12 images from each of A, B and C, with three from each batch in each time quarter. It includes 24 revised images and 12 expansion images. Selection used stage luminance and edge contrast, with at least 18 source frames between images. The reviewer then selected targets from raw overviews to include standing dancers, floor work, lifted poses, overlapping bodies, stage edges and low light. All images come from the same performance, and some late frames show the same final pose. [Sample and image hashes](sample.csv) · [Target selection](targets.csv)

## Blind comparison

The AI reviewer inspected raw images first, then measured boxes and points against an image-pixel grid. The observations were recorded and hashed before opening the merged annotation XML. An accepted match requires IoU of at least 0.50 and an advantage of at least 0.10 over the next candidate in that frame. Point distance is Euclidean distance in source pixels for accepted pairs with visible points in both annotations. [Frozen observations](blind_observations.csv) · [Per-frame results and source hashes](comparison.json)

| Review measure | Result |
|---|---:|
| Frames reviewed | 36 |
| Blind boxes | 35 |
| Accepted matches | 33 |
| Ambiguous matches | 2 |
| Abstentions before matching | 1 |
| Accepted box IoU, median / 10th percentile | 0.844 / 0.662 |
| Contact state agreement | 29 / 33 |
| Jointly visible point pairs | 26 |
| Support type agreement on those pairs | 24 / 26 |
| Point distance, median / 90th percentile | 5.0 / 13.7 px |
| Point pairs within 10 px | 22 / 26 |

The three dim-foreground disagreements were visible in the export and uncertain in the blind pass. The largest point gap, 37.1 px in frame 4980, came from choosing the palm versus the seat. In frame 3468, the 17.1 px gap exposed an error in the blind pass. The existing point is closer to the visible foot-floor junction. The comparison retains these original observations. The merged XML and training split retain their recorded hashes. [Post-comparison decisions](adjudication.csv) · [Overlay examples](review_examples.jpg)

## Reproduce and review

From the project root, run `python experiments/final_data_review/compare_audit.py`. The comparison uses the Python standard library. With Pillow installed, `prepare_audit.py` rebuilds the blank frame pack and `render_comparisons.py` rebuilds the example figure. Intermediate measurement sheets stay in `build/final_data_review`.

The blank pack lets actual reviewers annotate the 36 source images independently and compare full-frame labels after export. [Reviewer instructions](../../data/final_annotation_review/README.md) · [Blank review ZIP](../../data/final_annotation_review/raw_review_frames.zip)
