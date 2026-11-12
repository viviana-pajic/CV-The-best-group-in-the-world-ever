# Dance floor contact and formations

Start with the [master notebook](Master_Notebook.ipynb). It follows the project from labels and model comparisons to the 12 × 12 m map. The [research notebook](Research_Notebook.ipynb) contains the detailed evidence. Open [Formation Review](dashboard/index.html) in a browser to switch among the four saved replays.

- [Grading checklist and remaining work](docs/rubric_and_gaps.md)
- [Annotation guide](Labeling_Guide.ipynb), [review packs](data/review_packs) and [Windows labeler](annotator/README.md)
- [Error casebook](docs/error_casebook.md), [experiment log](docs/experiment_log.md), [AI disclosure](docs/genai_disclosure.md) and [defense notes](docs/defense_notes.md)

## Get the project

Install Git LFS to download the videos, model weights, and annotation packs with the repository.

```powershell
git lfs install
git clone https://github.com/viviana-pajic/CV-The-best-group-in-the-world-ever.git
cd CV-The-best-group-in-the-world-ever
git lfs pull
```

## Run locally

Use Python 3.12. Install `requirements-notebooks.txt` to open the notebooks in JupyterLab. The saved outputs are included. To run inference, use a separate environment and install `requirements-app.txt`. A CUDA build of PyTorch 2.11 and torchvision 0.26 is required to use the RTX path. See [setup and reproduction](docs/reproduction.md).

```powershell
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
python scripts/run_demo.py --output build/fresh_demo
python tools/evaluate_submission_detector.py --output build/detector_recheck
```

The app accepts constant-rate 16:9 MP4 clips up to three minutes. It can align the original venue automatically or accept four visible stage corners for a fixed camera. It exports a synchronized MP4, contacts CSV, frame JSON, calibration and run metadata.
