# Product validation and defense rehearsal

This is an AI-assisted rehearsal on the extracted local submission. Teammate participation and checks on teammates' computers are still pending.

## Four-minute defense walkthrough

1. **Explain the judging task.** Formation Review helps a dance judge assess alignment within a detected row. Floor contacts locate dancers on the stage map. The judge supplies context about the intended formation.
2. **Show a fitted row.** Select onare and pause at **03:04.68**, frame **4,617**. With the default four-dot minimum and ±10° angle limit, one line uses six of the seven displayed dots and scores **91.7%** at **−5.5°**. Move the maximum angle to horizontal to show the score fall to **83.4%**. Raising the minimum to ten yields an empty score because the clip has too few dots for that setting. Restore the defaults after the example.
3. **Show stage use.** Select Dernek and seek to the end. Detected coverage reaches **123 of 144 stage squares (85.4%)**. Rewind to frame zero to show the count return to zero. The map accumulates reported floor contacts through the playhead and uses a relative color scale. The grid count records which one-meter squares received a reported point.
4. **Show one limitation and one export.** At **00:17.88** in Wicked Games the dashboard has only **two dots**. Use this moment to explain how visibility affects the positions available to a judge. In the upload app, choose CPU and upload the bundled 16-frame example at `experiments/integration/cpu_smoke/input.mp4`. Show the synchronized replay and the CSV with frame, timestamp, and stage coordinates.

## Questions to practice

| Perspective | Question | Answer to support with the demo |
|---|---|---|
| Data | Why can a dancer disappear from the fit? | Occlusion and low light can prevent a usable reported contact. Explain the visible contact labeling policy and show Wicked Games at 00:17.88. |
| Model | What does 91.7% measure? | Mean perpendicular distance from the six reported positions to their fitted line, converted to an alignment score. Explain how the angle limit and minimum dancer count change the eligible rows. |
| Product | How do you verify synchronization and stage coverage? | Step one frame and inspect the shared playhead. Compare the MP4 frame count and FPS with the export. Rewind the heat map and explain its cumulative count. |

Before presenting, the actual team should try the walkthrough on the presentation machine, record what each member contributed, and confirm the sharing rights and course AI declaration.

## Checks completed

| Check | Result |
|---|---|
| Incoming package and standalone dashboard | Extracted the full ZIP and matched all **1,073** manifest hashes. The standalone ZIP's **18** dashboard files match the full package. This check covered the incoming ZIP before the upload fix. The final rebuilt ZIP needs its own checksum verification. [Archive results](archive_results.json) |
| Dashboard browser | Exercised the four-clip picker, onare's fit controls, Dernek playback and backward scrubbing, and a low-visibility moment in Wicked Games. Browser playback and the heat map both showed **00:00.40** after pausing Dernek. [Browser results](browser_results.json) |
| Heat map calculations | Independently counted reported contacts and visited cells for all four clips across forward and backward seeks. End counts were **102** for the original, **100** for Wicked Games, **103** for onare, and **123** for Dernek. The dashboard's smoothed grid preserved the sample totals. [Heat map checks](check_heatmap.cjs) · [Results](heatmap_results.json) |
| Fresh CPU upload | Streamlit processed the **16-frame**, **25 FPS** clip and produced **288** contact rows across **16** calibrated frames. All five export downloads were present. Replacing the input clears stale downloads. A corrupt MP4 and invalid manual corner input show errors. [Upload check](check_upload.py) · [Results](upload_results.json) |
| Existing project and annotator tests | Four geometry, split, timing, and export tests passed. Both annotator source tests passed, including a CVAT export round trip. The executable's bytes were included in the archive integrity check. |
| Local launcher | Verified loopback binding, HTML and JavaScript serving, exact MP4 byte ranges for seeking, and rejection of paths outside the dashboard. Browser automation blocked `file://`, so direct file playback remains unverified. Use the localhost launcher for the rehearsed path. [Launcher](open_dashboard.py) · [Results](launcher_results.json) |

The upload ran in the extracted project with the existing Python 3.12.10 environment and CUDA disabled. This demonstrates that the archive contains the required project files. Installation on a different machine remains a team check.
