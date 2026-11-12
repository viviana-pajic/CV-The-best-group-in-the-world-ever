"""Verify the labeling workflow and compatibility with the project dataset."""
from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image

from annotator import app
from annotator.annotation_io import export_xml, import_xml, new_project
from Experiment.Dance_Overnight.code.import_cvat_review import convert


class AnnotationWorkflowTest(unittest.TestCase):
    def test_draw_copy_recover_and_export(self):
        root = tk.Tk()
        self.addCleanup(root.destroy)
        root.update()
        ui = app.Labeler(root)
        errors = []
        root.report_callback_exception = lambda _, error, __: errors.append(error)
        root.geometry("1040x670")
        root.update()
        for control in (ui.redo_button, ui.floor_toggle, ui.fit_button):
            self.assertTrue(control.winfo_ismapped(), control)
            self.assertGreater(control.winfo_height(), 12, control)

        with TemporaryDirectory() as tmp, patch.object(app, "APP_DATA", Path(tmp) / "local"):
            folder = Path(tmp) / "sample images"
            folder.mkdir()
            for name in ("frame_000001.png", "frame_000002.png"):
                Image.new("RGB", (640, 360), "#334655").save(folder / name)
            ui.load_folder(folder)
            root.update()

            def event(action, x, y):
                sx, sy = ui.canvas._to_screen(x, y)
                ui.canvas.event_generate(action, x=round(sx), y=round(sy))
                root.update()

            def box(start, end):
                ui.set_tool("box")
                event("<ButtonPress-1>", *start)
                event("<B1-Motion>", *end)
                event("<ButtonRelease-1>", *end)

            box((180, 40), (260, 190))
            self.assertEqual(ui.selection, ("dancer", 0))
            ui.set_tool("point")
            event("<ButtonPress-1>", 220, 182)
            event("<ButtonRelease-1>", 220, 182)
            ui.support_var.set("Knees")
            ui.support_changed(None)
            self.assertEqual(ui.frame["dancers"][0]["support_type"], "knees")

            ui.set_tool("floor")
            for x, y in [(80, 300), (280, 245), (480, 300)]:
                event("<ButtonPress-1>", x, y)
                event("<ButtonRelease-1>", x, y)
            ui.finish_guide()
            self.assertEqual(len(ui.frame["floor_guides"][0]["points"]), 3)
            with patch.object(app.messagebox, "askyesno", return_value=True):
                ui.copy_floor()
            self.assertEqual(len(ui.project.frames[1]["floor_guides"]), 1)
            ui.undo()
            self.assertEqual(ui.project.frames[1]["floor_guides"], [])
            ui.redo()
            self.assertEqual(len(ui.project.frames[1]["floor_guides"]), 1)

            ui.navigate(1)
            ui.frame_split.set("Validation")
            ui.split_changed(None)
            self.assertTrue(ui.flush_save())
            ui.load_folder(folder)
            root.update()
            self.assertEqual(ui.index, 1)
            self.assertEqual(ui.canvas.sample_split, "val")
            box((270, 80), (370, 210))
            self.assertEqual(ui.frame["dancers"][0]["sample_split"], "val")

            output = Path(tmp) / "output.xml"
            with patch.object(app.filedialog, "asksaveasfilename", return_value=str(output)):
                ui.export()
            roundtrip = import_xml(new_project(folder, app.read_images(folder)), output)
            for actual, expected in zip(roundtrip.frames, ui.project.frames, strict=True):
                self.assertEqual(actual["dancers"], expected["dancers"])
                self.assertEqual(actual["floor_guides"], expected["floor_guides"])
            self.assertTrue(ui.flush_save())
            self.assertEqual(errors, [])

    def test_existing_cvat_dataset_round_trip(self):
        base = Path(__file__).resolve().parents[1] / "Experiment/lighting_expansion_200"
        images = base / "frames"
        revision = base / "visual_relabel"
        empty = new_project(images, app.read_images(images))
        original = import_xml(empty, revision / "cvat_visual_revision_export.zip")
        self.assertEqual(len(original.frames), 200)
        self.assertEqual(sum(len(f["dancers"]) for f in original.frames), 3689)
        self.assertEqual(sum(len(f["floor_guides"]) for f in original.frames), 2400)

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "annotations.xml"
            export_xml(original, output)
            self.assertEqual(import_xml(empty, output), original)
            self.assertEqual(convert(output), convert(revision / "verified_cvat_export.xml"))
