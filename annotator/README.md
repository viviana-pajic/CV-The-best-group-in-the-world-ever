# Floor Contact Labeler

Open **FloorContactLabeler.exe** on a 64-bit Windows PC. It runs locally and includes its Python and image libraries. Share the executable and the image folder with reviewers.

1. Choose **Open image folder**. The queue sorts numbered filenames in numeric order.
2. Press **B** and drag around each dancer's visible body. Each box receives an ID. Use **Set ID** to match an existing dancer ID when needed.
3. With a dancer selected, press **P** and click where their weight meets the floor. Use **Support type** for feet, knees, knee and foot, seat, or palms. Set the contact to **Hidden**, **Airborne**, or **Uncertain** when a visible point cannot be labeled.
4. Press **G**, choose a floor guide type, and click along the line. Press **Enter** to finish. For fixed-camera footage, use **Copy guides to empty frames** after tracing the floor. It copies only to images with matching dimensions and no existing guides.
5. Choose **Export CVAT XML** and save the annotations. Keep the original image files with the XML. The project importer uses source filenames such as `frame_003817.png`. Other image names work in the labeler.

Press **V** to select and edit a shape, **A / D** to switch images, **H** to pan, and **F** to fit the image. Scroll to zoom at the pointer. **Ctrl+Z / Ctrl+Y** undo and redo edits.

The app saves local drafts as you work. Reopen the same image folder to resume at the last image. Use **Project → Save project as** to share an editable draft. On another PC, open the JSON and locate the image folder if prompted.

To review the project's existing labels, open [the 200 training images](../Experiment/lighting_expansion_200/frames) and use **Project → Import CVAT XML or ZIP** to load [the verified CVAT export](../Experiment/lighting_expansion_200/visual_relabel/cvat_visual_revision_export.zip). Import replaces annotations for the matching images and can be undone.

## Build and test

From the repository root on 64-bit Windows with Python 3.12:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_annotator.ps1
.\build\annotator\.venv\Scripts\python.exe -m unittest annotator.test_annotator -v
```

The standalone executable is written to `build/annotator/dist/FloorContactLabeler.exe`. The tests use a Windows desktop session and the repository's CVAT sample.
