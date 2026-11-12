# Dance floor contact: reviewer checklist

Use your A, B or C pack. Start with the flagged frames below, then review every image in your pack. The frame filename is the reliable identifier; CVAT frame numbers assume the supplied image order.

1. **Boxes:** Keep one tight rectangle for each distinguishable dancer. Split merged boxes, remove duplicates, and add clearly missed dancers. Do not draw boxes for people you cannot distinguish.
2. **Status:** On the rectangle, set visible if you can locate the primary support, hidden if it is blocked, airborne if the dancer is clearly off the floor, or uncertain if the evidence is unclear.
3. **Point:** Visible means exactly one floor_support point at the actual floor contact. Use the planted sole, supporting knee(s), knee and foot midpoint, seat, or supporting palm(s) as appropriate. For two clearly planted supports, use their midpoint; for a single planted support, use that contact. Remove points for hidden, airborne, or uncertain cases.
4. **Link and split:** The rectangle and point must share a positive instance_id unique within the image. Replace the default 0 on new shapes. Keep sample_split on the rectangle; new rectangles inherit the split only when you set it to match the image. Validation images stay out of training.
5. **Finish:** Save and export CVAT for images 1.1. Run the supplied export check. Hand back the XML with your reviewer name. Resolve missing or orphaned points, duplicate IDs, and changed splits before merging A/B/C and retraining.

## Reviewer A

Start with frame_003290.png (CVAT frame 0): recheck the overlapping dancers in white and brown. Make sure there are separate boxes and each point belongs to the correct box. Then check frame_003640.png (validation, CVAT frame 4) for duplicate or merged boxes around overlaps.

Validation: frame_003640.png, frame_004585.png, frame_004615.png, frame_005510.png.

Training: frame_003318.png, frame_003332.png, frame_003493.png, frame_003507.png, frame_003843.png, frame_003857.png, frame_004143.png, frame_004157.png, frame_004179.png, frame_004221.png, frame_004343.png, frame_004357.png, frame_004420.png, frame_004456.png, frame_004492.png, frame_004528.png, frame_004645.png, frame_005343.png, frame_005357.png, frame_005420.png.

## Reviewer B

Start with frame_005371.png (CVAT frame 20), instance_id 8: the draft marks a seated contact at (591, 490). Confirm that the seat contact is visible. If the seat is obscured, mark hidden; if it is ambiguous, mark uncertain. Remove the point in either case.

Validation: frame_003682.png, frame_004595.png, frame_004705.png, frame_005550.png.

Training: frame_003346.png, frame_003360.png, frame_003521.png, frame_003535.png, frame_003871.png, frame_003885.png, frame_004171.png, frame_004185.png, frame_004193.png, frame_004235.png, frame_004371.png, frame_004385.png, frame_004432.png, frame_004468.png, frame_004504.png, frame_004540.png, frame_004665.png, frame_005371.png, frame_005385.png, frame_005440.png.

## Reviewer C

Start with frame_004207.png (CVAT frame 10), instance_id 5: verify the foot point at (409, 479) is on the actual planted support. Then check frame_005480.png (CVAT frame 22), instance_id 1: verify the seated point at (466, 453). If contact is obscured or ambiguous, change status and remove the point.

Validation: frame_003710.png, frame_004605.png, frame_004719.png.

Training: frame_003290.png, frame_003304.png, frame_003465.png, frame_003479.png, frame_003815.png, frame_003829.png, frame_004115.png, frame_004129.png, frame_004165.png, frame_004207.png, frame_004315.png, frame_004329.png, frame_004444.png, frame_004480.png, frame_004516.png, frame_004552.png, frame_005315.png, frame_005329.png, frame_005400.png, frame_005480.png.
