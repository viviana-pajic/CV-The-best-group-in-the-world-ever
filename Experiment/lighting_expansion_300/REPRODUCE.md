# Reproduce the 300-frame trial

Use the project checkout and a CUDA Python 3.12 environment with the [ML requirements](../Dance_Overnight/requirements.txt). The original runs used an RTX 5080 and PyTorch with CUDA 12.8.

1. From the project folder, run `python tools/build_300_packs.py`.
2. Create an empty `build/lighting_expansion_300` working directory. Copy the Python scripts, `assignments.json`, `training_manifest.json`, `detector_dev_references.json` and `merged_annotations.xml` from `Experiment/lighting_expansion_300` into it.
3. Extract the combined ZIP from `build/300-packs` and put its 300 training PNGs in `build/lighting_expansion_300/frames`. The training scripts read the original validation images separately from the checkout.
4. Run these commands from the project folder in PowerShell.

```powershell
$p = 'build/lighting_expansion_300'
python "$p/verify_dataset.py" --assignment-manifest "$p/assignments.json" --part "A=build/300-packs/Batch_A_100.zip" --part "B=build/300-packs/Batch_B_100.zip" --part "C=build/300-packs/Batch_C_100.zip" --merged "$p/merged_annotations.xml" --image-dir "$p/frames" --training-manifest "$p/training_manifest.json" --report "$p/structural_qa.json"
python "$p/train_300.py" support
python "$p/train_300.py" detector
python "$p/evaluate_300.py" support
python "$p/evaluate_300.py" detector
python "$p/evaluate_300.py" joint
```

The support comparison uses the published detections and tracks. The joint run rebuilds tracks with the new detector. Both use the published map fit, the same reserved images, and thresholds selected on validation. Each evaluation saves a 0.30 replay and its map coordinates.

The proposal and merge scripts document how the dataset was assembled. Regenerating proposals requires the local YOLO26m pose checkpoint and the previous trial's predictions. Training from the verified manifest uses the exported annotations.
