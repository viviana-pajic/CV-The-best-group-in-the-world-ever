"""Local upload and export interface for the floor-contact pipeline."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import run

st.set_page_config(page_title="Dance Floor Contact", layout="wide")
st.title("Dance Floor Contact")
st.write("Upload a fixed-camera MP4 to see the dancer detections and the synchronized 12 × 12 m stage map.")

source_choice = st.radio("Video source", ["Bundled sample", "Upload an MP4"], horizontal=True)
upload = st.file_uploader("Fixed-camera clip", type=["mp4"]) if source_choice == "Upload an MP4" else None
if upload:
    st.video(upload)
with st.expander("Calibration and device"):
    method = st.radio("Stage calibration", ["Match the original venue", "Enter four floor corners"])
    corner_text = ""
    if method == "Enter four floor corners":
        corner_text = st.text_area("Corner pixels as JSON", placeholder="[[320, 390], [960, 390], [1200, 610], [80, 610]]",
                                   help="Enter back-left, back-right, front-right, front-left in the source image.")
    device = st.radio("Device", ["Auto", "CPU"], horizontal=True)

if st.button("Run fresh inference", type="primary", disabled=source_choice == "Upload an MP4" and upload is None):
    st.session_state.pop("result", None)
    progress = st.progress(0, text="Checking the video")
    try:
        corners = json.loads(corner_text) if method == "Enter four floor corners" else None
        with TemporaryDirectory(prefix="dance-floor-contact-") as directory:
            workspace = Path(directory)
            source = ROOT / "media/sample_static.mp4"
            if upload:
                source = workspace / "upload.mp4"
                source.write_bytes(upload.getbuffer())
            output = workspace / "result"
            metadata = run(source, output, device.lower(), corners,
                           lambda message, fraction: progress.progress(fraction, text=message))
            st.session_state.result = (metadata, {path.name: path.read_bytes() for path in output.iterdir()})
        progress.empty()
    except (OSError, ValueError, RuntimeError) as error:
        progress.empty()
        st.error(str(error))

if result := st.session_state.get("result"):
    metadata, files = result
    if not metadata["frames_with_boxes"]:
        st.warning("No dancers were detected in this clip.")
    st.video(files["annotated_replay.mp4"])
    st.caption(f"{metadata['frames']:,} frames · {metadata['fps']:.2f} fps · {metadata['device']} · "
               f"{metadata['wall_seconds']:.1f} s processing")
    for label, filename, mime in (
        ("Annotated MP4", "annotated_replay.mp4", "video/mp4"),
        ("Contacts CSV", "contacts.csv", "text/csv"),
        ("Frame data JSON", "frames.json", "application/json"),
        ("Calibration JSON", "calibration.json", "application/json"),
        ("Run metadata", "metadata.json", "application/json"),
    ):
        st.download_button(label, files[filename], filename, mime, on_click="ignore")
