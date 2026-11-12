# Formation Review

Formation Review helps dance judges assess group formation accuracy. Fitted rows show how closely dancers line up on the stage map, and a live heat map shows stage use over time. Floor contacts provide the positions used in both views.

Extract the dashboard ZIP and open `index.html` in Chrome or Edge. Keep the `assets` folder beside it. The replay runs locally in the browser.

Use the timeline to pause on a formation. The arrow buttons move by one frame. Change **Minimum dots per line** and **Maximum line angle** to refit the same contacts. The initial settings are four contacts and up to 10° on either side of horizontal. The four display switches work independently.

The original replay covers source frames 3250–5673. Wicked Games, onare and Dernek cover their full source videos. All four replays use the same 300-image dancer and contact checkpoints and the 0.30 contact display threshold. The three additional clips use automatic tracking because they have different rosters. The video contact overlay projects the map dots back to the camera image using the stage calibration. Solid dots are reported contacts and hollow dots are temporal estimates. The line fitter uses solid contacts.

The stage is 12 × 12 m. The map and line fitter use the same scale on both axes. The additional clips align fixed venue landmarks to the original calibration as the camera zooms or shifts. Hover over a line score to see its average perpendicular offset in meters. Line scores describe the geometry of the displayed contacts. Measuring detector accuracy requires manually verified reference frames.

The live stage heat map accumulates reported contact time from the first frame of the selected clip through the playhead. Scrubbing back shows only the earlier portion of the clip. Color is relative to the busiest area so far. Detected coverage counts the 1 m squares reached by at least one reported contact. Missed contacts and occasional false detections can affect that count.

## Add another choreography

From the repository root, run `python tools/build_dashboard_data.py --help` with the project Python environment. Export the new clip using its source video, matching track and map exports, calibration and floor guides. Use a unique `--id` and a readable `--title`. For another clip on this stage, pass `--stage-meters 12 12`. Add a deferred script tag for the new `assets/<id>.js` in `index.html` next to the existing clip script. The selector lists every registered clip.

For another full MP4 on the same stage, the following commands produce the inference, camera alignment and dashboard assets. The inference command requires CUDA. Choose a distinct source filename stem to give the clip a distinct ID.

```powershell
python tools/run_external_replay.py path/to/clip.mp4
python tools/register_external_stage.py path/to/clip.mp4
python tools/project_external_replay.py path/to/clip.mp4 --title "Choreography name"
```
