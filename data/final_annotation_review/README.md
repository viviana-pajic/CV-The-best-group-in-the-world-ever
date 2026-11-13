# Independent annotation pack

The ZIP contains 36 raw images with the original pixel dimensions and a sample manifest. Each source batch contributes 12 images. The sample covers four time intervals, both dataset revisions and different image conditions. At least 18 source frames separate each pair of neighboring picks.

1. Extract `raw_review_frames.zip` and open its `images` folder in Floor Contact Labeler or CVAT. Create a fresh annotation project for each reviewer.
2. Draw one visible-extent box per distinguishable dancer and mark the sample split as train. Mark the contact as visible, hidden, airborne or uncertain. Add a point only when the floor contact is visible. Use the midpoint between two clear supports. Mark the support type as feet, knees, mixed knee and foot, palms or seat.
3. Record the reviewer identity, date, unclear cases and export filename. Export a separate XML for each pass. Open the comparison labels after both exports are complete.
4. Compare and adjudicate disagreements. Keep the original independent exports with the adjudicated version.

The pack has no preloaded annotations. These images come from the training set, so review measures annotation consistency on this sample. The full protocol is in `docs/annotation_policy.md` in the project.
