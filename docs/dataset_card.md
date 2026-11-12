# Dataset card

The source is one fixed-camera performance. The 300 training frames include the 200-frame visual revision and 100 new frames sampled across lighting and crowd conditions. The pilot used 60 training frames and 11 validation frames. A, B and C each contain 100 disjoint training frames.

The verified merged export has 5,459 dancer boxes and 4,068 visible support points. It includes 835 hidden, 59 airborne and 497 uncertain contacts. Uncertain entries retain boxes and were excluded from support training. Each training frame has 12 fixed floor guide segments. See the [exact dataset counts](../Experiment/lighting_expansion_300/DATASET.md), [frame manifest](../data/frame_manifest.csv) and [split audit](../experiments/data/split_audit.json).

The closest training and validation frames are 11 source frames apart. Source frames 3938–4092 are excluded from training, and seven frames within that block have reserved references. Exact training image hashes match the review records. Stage-crop luma comparison is a screen for similar images. All three labeled splits come from the same performance, and the reserved block was inspected in earlier iterations.

The annotation process used model proposals followed by assistant visual corrections. The three pack names are review assignments. The separate assistant repeat review covered 20 crops and provides within-workflow repeatability evidence. Independent team labels and agreement figures remain open.

The three later videos are used for inference and visual review. Their source links and sharing notes are in [footage and rights](rights.md).
