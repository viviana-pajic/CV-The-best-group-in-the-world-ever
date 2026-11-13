# Annotation and independent review

Draw one visible-extent box per distinguishable dancer. Include visible extremities and clip to the image boundaries. If no separate body can be distinguished during overlap, record the ambiguity without inventing a box. For blur, label the body only if it is identifiable and use **uncertain** if contact is ambiguous.

Use one positive ID per box within the image. A **visible** contact gets one matching point. Use the midpoint between two clear supports. A raised foot, an occluded support or an ambiguous shadow does not provide a point. Choose the relevant support type: feet, knees, mixed knee and foot, palms or seat. Mark hidden, airborne and uncertain separately. The floor guides describe the fixed reference plane.

For the independent audit, select 30–45 images across all batches, covering floor work, overlaps, low light, blur, edge positions and clear standing cases. Have two reviewers independently annotate the same raw frames before seeing the existing export or one another's work. Keep both CVAT XMLs and record who reviewed each and when. Pair visible bodies by one-to-one overlap, record unmatched boxes, then report median and lower-tail IoU, contact-state agreement, and point distance for jointly visible contacts. Adjudicate disagreements while preserving both originals and document which fixes enter any future training revision.

[Guide and illustrated examples](../Labeling_Guide.ipynb) · [Original protocol](../Experiment/Dance_Overnight/CVAT_Review/REVIEW_CHECKLIST.md) · [300-frame export audit](../Experiment/lighting_expansion_300/structural_qa.json)

The blank 36-frame pack is prepared for this independent review. A targeted AI pass on one dancer per frame supplies examples for adjudication after human annotations are recorded. [Blank pack and instructions](../data/final_annotation_review/README.md) · [AI audit](../experiments/final_data_review/REPORT.md)
