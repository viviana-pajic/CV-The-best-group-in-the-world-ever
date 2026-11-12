# 300-frame dance floor contact dataset

The training set combines 200 frames from the previous visual revision and 100 new source frames. The new frames were selected across the static-camera segment for variation in lighting and crowd conditions, with nearby labeled frames available for annotation proposals. Adjacent images from the same video remain correlated.

A, B, and C each contain 100 distinct training images. They are review assignments. The labels were produced with model-assisted proposals and assistant visual review. The previous 200 image files and labels were verified against the original revision. The new 100 were checked against the source image and overlays, with corrections for clipped limbs, missed dancers, duplicates, and support points. This is approximate visual annotation. Dense occlusion and dark frames can still hide dancers or contacts.

| Batch | Previous frames | New frames | Boxes | Visible points |
|---|---:|---:|---:|---:|
| A | 65 | 35 | 1,830 | 1,331 |
| B | 65 | 35 | 1,821 | 1,360 |
| C | 70 | 30 | 1,808 | 1,377 |
| Combined | 200 | 100 | 5,459 | 4,068 |

Each frame has 12 fixed-camera floor guides, including an estimated rear edge. Of the 5,459 dancer annotations, 835 contacts are hidden, 59 are airborne, and 497 remain uncertain. Uncertain entries retain their boxes and were excluded from support training.

The training manifest preserves the original 11 validation images and excludes their 10-frame guard and the reserved block at source frames 3938–4092. The three review exports merge exactly into the combined XML. `structural_qa.json` records the geometry, split, and checksum checks.

## Open a review batch in Floor Contact Labeler

1. Extract the batch ZIP.
2. Open its `images` folder in Floor Contact Labeler.
3. Choose **Project → Import CVAT XML or ZIP** and select `annotations.xml`.
4. Save a draft or export revised labels after review.

The combined ZIP includes the 300 training PNGs under `images`, the 11 validation PNGs under `validation_images`, and the exact training manifest. The validation PNGs should stay outside the training image folder.
