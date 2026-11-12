"""Local Windows editor for the dance floor contact annotation schema."""
from collections import deque
from copy import deepcopy
import ctypes
import hashlib
import logging
import os
from pathlib import Path
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, UnidentifiedImageError

from annotator.annotation_io import (
    CONTACTS, SPLITS, SUPPORT_TYPES,
    export_xml, import_xml, load_project, new_project, save_project,
)
from annotator.canvas import AnnotationCanvas

APP_NAME = "Floor Contact Labeler"
APP_DATA = Path(os.getenv("LOCALAPPDATA", Path.home())) / "DanceFloorContactLabeler"
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

BG = "#0f1421"
PANEL = "#171e2d"
PANEL_HOVER = "#253147"
FIELD = "#101724"
BORDER = "#2d3950"
INK = "#edf2fc"
MUTED = "#a4b1c8"
ACCENT = "#79f0cd"
ACCENT_HOVER = "#a0f7de"
ON_ACCENT = "#102c28"
PURPLE = "#c4b5fd"
DANGER = "#ff9d9d"
FONT = "Segoe UI"

TOOL_HINTS = {
    "select": "Select a shape to move it or drag its handles.",
    "box": "Drag around the dancer's visible body.",
    "point": "Select a dancer, then click the weight-bearing point.",
    "floor": "Click to place vertices. Press Enter to finish the guide.",
    "pan": "Drag the image. Scroll to zoom around the pointer.",
}
SUPPORT_NAMES = {"none": "None", "feet": "Feet", "knees": "Knees", "mixed_knee_foot": "Knee and foot",
                 "seat": "Seat", "palms": "Palms"}
FLOOR_NAMES = {"mat_boundary": "Mat boundary", "mat_rear_estimate": "Estimated rear edge", "mat_grid": "Grid line"}
SPLIT_NAMES = {"train": "Training", "val": "Validation", "reserved_test": "Reserved test", "unset": "Unset"}


def read_images(folder):
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES]
    paths.sort(key=lambda p: tuple(int(s) if s.isdigit() else s.casefold() for s in re.split(r"(\d+)", p.name)))
    if not paths:
        raise ValueError("Choose a folder containing PNG, JPG, BMP, WebP, or TIFF images.")
    records = []
    for path in paths:
        try:
            with Image.open(path) as im:
                records.append((path.name, *im.size))
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            raise ValueError(f"Could not read {path.name}. {error}") from error
    return records


def configure_style(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=PANEL, foreground=INK, font=(FONT, 10))
    style.configure("TButton", padding=(12, 7), borderwidth=1, bordercolor=BORDER,
                    lightcolor=PANEL, darkcolor=PANEL, focuscolor=ACCENT)
    style.map("TButton", background=[("active", PANEL_HOVER), ("pressed", FIELD)],
              foreground=[("disabled", MUTED)])
    style.configure("Accent.TButton", background=ACCENT, foreground=ON_ACCENT, bordercolor=ACCENT,
                    lightcolor=ACCENT, darkcolor=ACCENT, font=(FONT, 10, "bold"))
    style.map("Accent.TButton", background=[("disabled", PANEL_HOVER), ("active", ACCENT_HOVER)],
              foreground=[("disabled", MUTED), ("!disabled", ON_ACCENT)])
    style.configure("Tool.TButton", padding=(8, 6), width=0)
    style.configure("Compact.TButton", padding=(8, 4), width=0)
    style.configure("Selected.Tool.TButton", background="#284740", foreground=ACCENT,
                    bordercolor="#528f7f", padding=(8, 6), font=(FONT, 10, "bold"))
    style.map("Selected.Tool.TButton", background=[("active", "#345b51")])
    style.configure("TCombobox", padding=(7, 3), arrowcolor=MUTED, bordercolor=BORDER,
                    lightcolor=FIELD, darkcolor=FIELD, selectbackground=FIELD, selectforeground=INK)
    style.map("TCombobox", fieldbackground=[("disabled", FIELD), ("readonly", FIELD)],
              foreground=[("disabled", MUTED), ("readonly", INK)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", INK)])
    style.configure("Treeview", background=FIELD, fieldbackground=FIELD, foreground=INK,
                    rowheight=30, borderwidth=0, bordercolor=FIELD,
                    lightcolor=FIELD, darkcolor=FIELD, font=(FONT, 9))
    style.map("Treeview", background=[("selected", "#284740")], foreground=[("selected", ACCENT)])
    style.configure("Vertical.TScrollbar", background=PANEL_HOVER, troughcolor=FIELD,
                    bordercolor=PANEL, lightcolor=PANEL_HOVER, darkcolor=PANEL_HOVER, arrowcolor=MUTED)
    style.configure("TCheckbutton", background=PANEL, foreground=MUTED, padding=(3, 3), indicatorcolor=FIELD)
    style.map("TCheckbutton", background=[("active", PANEL)], indicatorcolor=[("selected", ACCENT)])
    style.configure("Horizontal.TProgressbar", thickness=5, troughcolor=FIELD, background=ACCENT,
                    bordercolor=FIELD, lightcolor=ACCENT, darkcolor=ACCENT)
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", INK)
    root.option_add("*TCombobox*Listbox.selectBackground", "#284740")
    root.option_add("*TCombobox*Listbox.selectForeground", ACCENT)


def connect_scrollbar(tree, scrollbar):
    def show_scroll_position(first, last):
        scrollbar.set(first, last)
        if float(first) <= 0 and float(last) >= 1:
            scrollbar.pack_forget()
        elif not scrollbar.winfo_manager():
            scrollbar.pack(side="right", fill="y", before=tree)

    tree.configure(yscrollcommand=show_scroll_position)
    scrollbar.configure(command=tree.yview)


class Labeler:
    def __init__(self, root):
        self.root = root
        self.project = None
        self.index = 0
        self.selection = (None, None)
        self.history = deque(maxlen=80)
        self.redo_stack = deque(maxlen=80)
        self.dirty = False
        self.autosave_job = None
        self.tool = "select"
        self.guide_kind = tk.StringVar(value=FLOOR_NAMES["mat_boundary"])
        self.frame_split = tk.StringVar(value="Training")
        self.dancer_layer = tk.BooleanVar(value=True)
        self.floor_layer = tk.BooleanVar(value=True)
        root.title(APP_NAME)
        root.configure(bg=BG)
        root.geometry(f"{min(1440, root.winfo_screenwidth() - 48)}x{min(860, root.winfo_screenheight() - 110)}")
        root.minsize(1040, 670)
        configure_style(root)
        self.build_ui()
        self.bind_keys()
        self.refresh_controls()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.report_callback_exception = self.report_error

    @property
    def frame(self):
        return self.project.frames[self.index] if self.project else None

    def label(self, parent, text="", *, size=10, color=INK, bold=False, bg=PANEL, anchor="w", **kw):
        return tk.Label(parent, text=text, bg=bg, fg=color,
                        font=(FONT, size, "bold" if bold else "normal"), anchor=anchor, **kw)

    def button(self, parent, text, command, style="TButton", **kw):
        return ttk.Button(parent, text=text, command=command, style=style, **kw)

    def build_ui(self):
        root = self.root
        top = tk.Frame(root, bg=BG, padx=22, pady=15)
        top.pack(fill="x")
        tk.Label(top, text="FC", bg=ACCENT, fg=ON_ACCENT, width=3, height=2,
                 font=(FONT, 12, "bold")).pack(side="left", padx=(0, 12))
        identity = tk.Frame(top, bg=BG)
        identity.pack(side="left")
        self.label(identity, APP_NAME, size=19, bold=True, bg=BG).pack(fill="x")
        self.folder_title = self.label(identity, "Dance floor annotation", color=MUTED, bg=BG)
        self.folder_title.pack(fill="x")
        self.export_button = self.button(top, "Export CVAT XML", self.export, "Accent.TButton")
        self.export_button.pack(side="right")
        self.button(top, "Open image folder", self.open_folder).pack(side="right", padx=(0, 8))
        self.project_button = self.button(top, "Project  ▾", self.show_menu)
        self.project_button.pack(side="right", padx=(0, 8))
        self.button(top, "Shortcuts", self.help).pack(side="right", padx=(0, 8))
        self.build_menu()

        body = tk.Frame(root, bg=BG, padx=14)
        body.pack(fill="both", expand=True)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        left = tk.Frame(body, bg=PANEL, width=206, highlightthickness=1, highlightbackground=BORDER)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 9))
        left.pack_propagate(False)
        self.label(left, "Image queue", size=13, bold=True).pack(fill="x", padx=14, pady=(14, 2))
        self.progress_text = self.label(left, "No folder open", color=MUTED, size=9)
        self.progress_text.pack(fill="x", padx=14)
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", padx=14, pady=(8, 12))
        listing = tk.Frame(left, bg=FIELD)
        listing.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        scrollbar = ttk.Scrollbar(listing, orient="vertical")
        self.queue = ttk.Treeview(listing, columns=("marks",), show="tree", selectmode="browse")
        self.queue.column("#0", width=135, minwidth=55, stretch=True)
        self.queue.column("marks", width=30, minwidth=30, stretch=False, anchor="center")
        self.queue.tag_configure("marked", foreground=ACCENT)
        self.queue.pack(side="left", fill="both", expand=True)
        connect_scrollbar(self.queue, scrollbar)
        self.queue.bind("<<TreeviewSelect>>", self.queue_selected)

        center = tk.Frame(body, bg=BG)
        center.grid(row=0, column=1, sticky="nsew")
        tools = tk.Frame(center, bg=PANEL, padx=8, pady=6, highlightthickness=1, highlightbackground=BORDER)
        tools.pack(fill="x")
        self.tool_buttons = {}
        for key, title in (("select", "Select  V"), ("box", "Box  B"), ("point", "Support  P"),
                           ("floor", "Floor  G"), ("pan", "Pan  H")):
            button = self.button(tools, title, lambda name=key: self.set_tool(name), "Tool.TButton")
            button.pack(side="left", padx=(0, 4))
            self.tool_buttons[key] = button
        self.fit_button = self.button(tools, "Fit", self.fit, "Tool.TButton")
        self.fit_button.pack(side="right")
        self.workspace = tk.Frame(center, bg=FIELD, highlightthickness=1, highlightbackground=BORDER)
        self.workspace.pack(fill="both", expand=True, pady=8)
        self.canvas = AnnotationCanvas(self.workspace, self.canvas_changed, self.canvas_selected, self.set_status)
        self.canvas.pack(fill="both", expand=True)
        self.welcome = tk.Frame(self.workspace, bg=FIELD, padx=25, pady=25)
        self.label(self.welcome, "DANCE FLOOR ANNOTATION", color=ACCENT, bold=True, bg=FIELD).pack(anchor="w")
        self.label(self.welcome, "Open a folder to start", size=26, bold=True, bg=FIELD).pack(anchor="w", pady=(6, 8))
        self.label(self.welcome, "Draw dancer boxes, mark weight-bearing points, and trace the floor.",
                   color=MUTED, bg=FIELD, wraplength=420, justify="left").pack(anchor="w")
        welcome_buttons = tk.Frame(self.welcome, bg=FIELD)
        welcome_buttons.pack(anchor="w", pady=(20, 0))
        self.button(welcome_buttons, "Open image folder", self.open_folder, "Accent.TButton").pack(side="left")
        self.button(welcome_buttons, "Open saved project", self.open_project).pack(side="left", padx=(8, 0))
        self.welcome.place(relx=.5, rely=.5, anchor="center")
        navigation = tk.Frame(center, bg=PANEL, padx=8, pady=6, highlightthickness=1, highlightbackground=BORDER)
        navigation.pack(fill="x")
        self.prev_button = self.button(navigation, "‹  Previous", lambda: self.navigate(-1))
        self.prev_button.pack(side="left")
        self.next_button = self.button(navigation, "Next  ›", lambda: self.navigate(1))
        self.next_button.pack(side="right")
        self.frame_title = self.label(navigation, "No image selected", size=9, color=MUTED, anchor="center")
        self.frame_title.pack(side="left", fill="x", expand=True, padx=8)

        right = tk.Frame(body, bg=PANEL, width=254, highlightthickness=1, highlightbackground=BORDER)
        right.grid(row=0, column=2, sticky="ns", padx=(9, 0))
        right.pack_propagate(False)
        self.build_inspector(right)

        footer = tk.Frame(root, bg=BG, padx=20, pady=9)
        footer.pack(fill="x")
        self.status = self.label(footer, "Open a folder of images to begin.", size=9, color=MUTED, bg=BG)
        self.status.pack(side="left")
        self.save_status = self.label(footer, "Offline workspace", size=9, color=ACCENT, bg=BG)
        self.save_status.pack(side="right")

    def build_menu(self):
        self.project_menu = tk.Menu(self.root, tearoff=False, bg=PANEL, fg=INK,
                                    activebackground=PANEL_HOVER, activeforeground=ACCENT,
                                    font=(FONT, 10), borderwidth=0)
        self.project_menu.add_command(label="Open saved project…", command=self.open_project)
        self.project_menu.add_command(label="Save project as…", command=self.save_as, accelerator="Ctrl+S")
        self.project_menu.add_separator()
        self.project_menu.add_command(label="Import CVAT XML or ZIP…", command=self.import_annotations)

    def show_menu(self):
        button = self.project_button
        self.project_menu.entryconfigure(1, state="normal" if self.project else "disabled")
        self.project_menu.entryconfigure(3, state="normal" if self.project else "disabled")
        try:
            self.project_menu.tk_popup(button.winfo_rootx(), button.winfo_rooty() + button.winfo_height())
        finally:
            self.project_menu.grab_release()

    def section(self, parent, title):
        panel = tk.Frame(parent, bg=PANEL)
        panel.pack(fill="x", padx=13, pady=(0, 5))
        self.label(panel, title, color=PURPLE, size=9, bold=True).pack(fill="x", pady=(0, 3))
        return panel

    def combo(self, parent, var, values, handler):
        box = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=18)
        box.pack(fill="x", pady=(0, 4))
        box.bind("<<ComboboxSelected>>", handler)
        return box

    def build_inspector(self, right):
        history = tk.Frame(right, bg=PANEL)
        history.pack(side="bottom", fill="x", padx=13, pady=(0, 10))
        self.undo_button = self.button(history, "Undo", self.undo, "Compact.TButton")
        self.undo_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.redo_button = self.button(history, "Redo", self.redo, "Compact.TButton")
        self.redo_button.pack(side="left", fill="x", expand=True)

        self.label(right, "Annotations", size=13, bold=True).pack(fill="x", padx=13, pady=(10, 2))
        self.count_text = self.label(right, "0 dancers · 0 points · 0 guides", color=MUTED, size=9)
        self.count_text.pack(fill="x", padx=13, pady=(0, 6))
        selection = self.section(right, "SELECTION")
        chooser = tk.Frame(selection, bg=FIELD)
        chooser.pack(fill="x", pady=(0, 6))
        scrollbar = ttk.Scrollbar(chooser, orient="vertical")
        self.shapes = ttk.Treeview(chooser, show="tree", selectmode="browse", height=2)
        self.shapes.column("#0", width=175, stretch=True)
        self.shapes.pack(side="left", fill="x", expand=True)
        connect_scrollbar(self.shapes, scrollbar)
        self.shapes.bind("<<TreeviewSelect>>", self.shape_selected)
        self.selection_text = self.label(selection, "Select a dancer or floor guide.", color=MUTED, size=9)
        self.selection_text.pack(fill="x", pady=(0, 3))

        detail = self.section(right, "DANCER CONTACT")
        self.contact_var = tk.StringVar()
        self.support_var = tk.StringVar()
        self.contact_box = self.combo(detail, self.contact_var, [s.title() for s in CONTACTS], self.contact_changed)
        self.support_box = self.combo(detail, self.support_var, [SUPPORT_NAMES[s] for s in SUPPORT_TYPES], self.support_changed)
        row = tk.Frame(detail, bg=PANEL)
        row.pack(fill="x")
        self.id_button = self.button(row, "Set ID", self.set_id, "Compact.TButton")
        self.id_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.delete_button = self.button(row, "Delete", self.delete_selected, "Compact.TButton")
        self.delete_button.pack(side="left", fill="x", expand=True)

        floor = self.section(right, "FLOOR GUIDE")
        self.guide_box = self.combo(floor, self.guide_kind, list(FLOOR_NAMES.values()), self.guide_changed)
        self.copy_button = self.button(floor, "Copy guides to empty frames", self.copy_floor, "Compact.TButton")
        self.copy_button.pack(fill="x")

        frame = self.section(right, "FRAME SPLIT")
        self.split_box = self.combo(frame, self.frame_split, [SPLIT_NAMES[s] for s in SPLITS], self.split_changed)
        layers = tk.Frame(frame, bg=PANEL)
        layers.pack(fill="x")
        ttk.Checkbutton(layers, text="Dancers", variable=self.dancer_layer, command=self.toggle_layers).pack(side="left")
        self.floor_toggle = ttk.Checkbutton(layers, text="Floor guides", variable=self.floor_layer, command=self.toggle_layers)
        self.floor_toggle.pack(side="left")


    def bind_keys(self):
        for event, action in (
            ("<Control-o>", self.open_folder), ("<Control-s>", self.save_as),
            ("<Control-e>", self.export), ("<Control-z>", self.undo),
            ("<Control-y>", self.redo), ("<F1>", self.help),
        ):
            self.root.bind(event, lambda _, callback=action: self.key_action(callback))
        for event, action in (
            ("<a>", lambda: self.navigate(-1)), ("<d>", lambda: self.navigate(1)),
            ("<Prior>", lambda: self.navigate(-1)), ("<Next>", lambda: self.navigate(1)),
            ("<v>", lambda: self.set_tool("select")), ("<b>", lambda: self.set_tool("box")),
            ("<p>", lambda: self.set_tool("point")), ("<g>", lambda: self.set_tool("floor")),
            ("<h>", lambda: self.set_tool("pan")), ("<f>", self.fit),
            ("<Delete>", self.delete_selected),
            ("<Return>", self.finish_guide), ("<Escape>", self.cancel_edit),
        ):
            self.root.bind(event, lambda e, callback=action: self.key_action(callback)
                           if not isinstance(e.widget, (ttk.Combobox, tk.Entry, tk.Text)) else None)

    @staticmethod
    def key_action(callback):
        callback()
        return "break"

    def finish_guide(self):
        if self.tool == "floor":
            self.canvas.finish_floor()

    def cancel_edit(self):
        if self.canvas.cancel():
            self.set_status("Drawing canceled.")

    def set_status(self, text):
        self.status.configure(text=text)

    def report_error(self, typ, value, trace):
        logging.error("Unhandled GUI callback", exc_info=(typ, value, trace))
        self.show_error("The action could not be completed", value)

    def show_error(self, title, error):
        messagebox.showerror(title, str(error), parent=self.root)

    def help(self):
        messagebox.showinfo(
            "Labeling shortcuts",
            "V   Select or resize a shape\n"
            "B   Draw a dancer box\n"
            "P   Place the selected dancer's support point\n"
            "G   Trace a floor guide, then press Enter\n"
            "H   Pan the image\n"
            "F   Fit the image\n"
            "A / D   Previous / next image\n"
            "Scroll   Zoom at the pointer\n"
            "Esc   Cancel the current drawing\n"
            "Delete   Delete the selected annotation\n"
            "Ctrl+Z / Ctrl+Y   Undo / redo\n"
            "Ctrl+S   Save a shareable project draft\n"
            "Ctrl+E   Export CVAT XML\n\n"
            "For a visible contact, click the point bearing the dancer's weight, then choose its type. "
            "Use Hidden, Airborne, or Uncertain if the point cannot be labeled. "
            "For a fixed camera, trace guides once and copy them to empty frames.",
            parent=self.root,
        )

    def draft_path(self, folder):
        digest = hashlib.sha256(os.path.normcase(str(folder.resolve())).encode("utf-8")).hexdigest()
        return APP_DATA / "drafts" / f"{digest}.json"

    @staticmethod
    def image_signature(project):
        return [(f["name"], f["width"], f["height"]) for f in project.frames]

    def open_folder(self):
        chosen = filedialog.askdirectory(title="Choose the folder containing your images", parent=self.root)
        if chosen:
            self.load_folder(Path(chosen))

    def load_folder(self, folder):
        if not self.leave_project():
            return
        try:
            folder = folder.resolve()
            records = read_images(folder)
            project = new_project(folder, records)
            draft = self.draft_path(folder)
            if draft.is_file():
                try:
                    recovered = load_project(draft)
                    if self.image_signature(recovered) != records:
                        raise ValueError("The list of images or their dimensions changed.")
                    recovered.image_dir = folder
                    project = recovered
                except (ValueError, OSError) as error:
                    self.show_error("Could not recover the local draft", f"{error}\n\nUse Project → Open saved project to recover a separate draft.")
                    return
            self.install_project(project)
            self.set_status(f"Opened {len(records):,} images. Local recovery is enabled.")
        except (OSError, ValueError) as error:
            self.show_error("Could not open the image folder", error)

    def open_project(self):
        path = filedialog.askopenfilename(title="Open a labeling project",
                                          filetypes=[("Labeling project", "*.json")], parent=self.root)
        if not path or not self.leave_project():
            return
        try:
            project = load_project(Path(path))
            if not all((project.image_dir / f["name"]).is_file() for f in project.frames):
                folder = filedialog.askdirectory(title="Locate this project's image folder", parent=self.root)
                if not folder:
                    return
                project.image_dir = Path(folder).resolve()
            available = {name: (width, height) for name, width, height in read_images(project.image_dir)}
            for frame in project.frames:
                if available.get(frame["name"]) != (frame["width"], frame["height"]):
                    raise ValueError(f"Missing image or dimension mismatch: {frame['name']}")
            self.install_project(project)
            self.schedule_save()
            self.set_status(f"Opened {Path(path).name}.")
        except (OSError, ValueError) as error:
            self.show_error("Could not open the saved project", error)

    def install_project(self, project):
        self.project = project
        self.history.clear()
        self.redo_stack.clear()
        self.welcome.place_forget()
        self.folder_title.configure(text=project.image_dir.name)
        self.queue.delete(*self.queue.get_children())
        for index, frame in enumerate(project.frames):
            self.queue.insert("", "end", iid=str(index), text=frame["name"])
            self.refresh_queue_row(index)
        self.dirty = False
        self.save_status.configure(text="Local draft ready", fg=ACCENT)
        resume_index = next((i for i, frame in enumerate(project.frames)
                             if frame["name"] == project.last_image), 0)
        self.show_frame(resume_index)
        self.refresh_progress()

    def show_frame(self, index):
        self.canvas.cancel()
        self.index = index
        self.selection = (None, None)
        frame = self.frame
        self.set_split_for_frame()
        self.canvas.set_frame(self.project.image_dir / frame["name"], frame["dancers"], frame["floor_guides"])
        self.canvas.set_visibility(dancers=self.dancer_layer.get(), floor=self.floor_layer.get())
        self.queue.selection_set(str(index))
        self.queue.see(str(index))
        self.frame_title.configure(text=f"{index + 1} / {len(self.project.frames)}    ·    {frame['name']}    ·    "
                                       f"{frame['width']} × {frame['height']}")
        self.refresh_shapes()
        self.refresh_controls()
        if self.project.last_image != frame["name"]:
            self.project.last_image = frame["name"]
            self.schedule_save()

    def navigate(self, delta):
        if not self.project:
            return
        target = self.index + delta
        if 0 <= target < len(self.project.frames):
            self.show_frame(target)

    def queue_selected(self, _):
        selected = self.queue.selection()
        if selected and int(selected[0]) != self.index:
            self.show_frame(int(selected[0]))

    def refresh_queue_row(self, index):
        frame = self.project.frames[index]
        dancer_count = len(frame["dancers"])
        guide_count = len(frame["floor_guides"])
        self.queue.item(str(index), values=(str(dancer_count) if dancer_count else "•" if guide_count else "",),
                        tags=("marked",) if dancer_count else ())

    def refresh_progress(self):
        if self.project:
            total = len(self.project.frames)
            completed = sum(bool(frame["dancers"]) for frame in self.project.frames)
            self.progress.configure(maximum=total, value=completed)
            self.progress_text.configure(text=f"{completed:,} / {total:,} with dancer labels")

    def refresh_shapes(self):
        frame = self.frame
        self.shapes.delete(*self.shapes.get_children())
        if not frame:
            return
        for index, row in enumerate(frame["dancers"]):
            point = "  ·  point" if row["point"] is not None else ""
            self.shapes.insert("", "end", iid=f"dancer:{index}",
                               text=f"Dancer {row['id']}  ·  {row['contact'].title()}{point}")
        for index, guide in enumerate(frame["floor_guides"]):
            self.shapes.insert("", "end", iid=f"floor:{index}", text=FLOOR_NAMES[guide["label"]])
        self.sync_shape_selection()
        self.count_text.configure(text=f"{len(frame['dancers'])} dancers · "
                                  f"{sum(row['point'] is not None for row in frame['dancers'])} points · "
                                  f"{len(frame['floor_guides'])} guides")

    def sync_shape_selection(self):
        kind, index = self.selection
        key = f"{kind}:{index}"
        if self.shapes.exists(key):
            self.shapes.selection_set(key)
            self.shapes.see(key)
        elif self.shapes.selection():
            self.shapes.selection_remove(*self.shapes.selection())

    def refresh_controls(self):
        frame = self.frame
        kind, index = self.selection
        selected_dancer = frame["dancers"][index] if frame and kind == "dancer" and index < len(frame["dancers"]) else None
        selected_guide = frame["floor_guides"][index] if frame and kind == "floor" and index < len(frame["floor_guides"]) else None
        self.selection_text.configure(text=f"Dancer {selected_dancer['id']} selected" if selected_dancer else
                                      "Floor guide selected" if selected_guide else "Select a dancer or floor guide.")
        self.contact_var.set(selected_dancer["contact"].title() if selected_dancer else "Contact status")
        self.support_var.set(SUPPORT_NAMES[selected_dancer.get("support_type") or "none"] if selected_dancer else "Support type")
        self.contact_box.configure(state="readonly" if selected_dancer else "disabled")
        self.support_box.configure(state="readonly" if selected_dancer and selected_dancer["point"] is not None else "disabled")
        self.id_button.configure(state="normal" if selected_dancer else "disabled")
        self.delete_button.configure(state="normal" if selected_dancer or selected_guide else "disabled")
        self.guide_box.configure(state="readonly" if frame else "disabled")
        self.copy_button.configure(state="normal" if frame and frame["floor_guides"] else "disabled")
        self.split_box.configure(state="readonly" if frame else "disabled")
        self.export_button.configure(state="normal" if frame else "disabled")
        self.prev_button.configure(state="normal" if frame and self.index else "disabled")
        self.next_button.configure(state="normal" if frame and self.index + 1 < len(self.project.frames) else "disabled")
        self.undo_button.configure(state="normal" if self.history else "disabled")
        self.redo_button.configure(state="normal" if self.redo_stack else "disabled")
        for name, button in self.tool_buttons.items():
            button.configure(state="normal" if frame else "disabled",
                             style="Selected.Tool.TButton" if self.tool == name else "Tool.TButton")

    def shape_selected(self, _):
        selected = self.shapes.selection()
        if selected:
            kind, index = selected[0].split(":")
            value = kind, int(index)
            if value != self.selection:
                self.canvas.select(*value)
                self.canvas_selected(*value)

    def canvas_selected(self, kind, index):
        self.selection = kind, index
        if kind == "floor" and self.frame and index < len(self.frame["floor_guides"]):
            label = self.frame["floor_guides"][index]["label"]
            self.guide_kind.set(FLOOR_NAMES[label])
            self.canvas.set_floor_label(label)
        self.sync_shape_selection()
        self.refresh_controls()

    def canvas_changed(self, dancers, floor_guides):
        updated = {**self.frame, "dancers": deepcopy(dancers), "floor_guides": deepcopy(floor_guides)}
        self.apply({self.index: updated}, update_canvas=False)

    def set_tool(self, tool):
        if not self.frame:
            return
        self.tool = tool
        if tool in {"box", "point"} and not self.dancer_layer.get():
            self.dancer_layer.set(True)
            self.toggle_layers()
        if tool == "floor" and not self.floor_layer.get():
            self.floor_layer.set(True)
            self.toggle_layers()
        self.canvas.set_tool(tool)
        self.canvas.focus_set()
        self.set_status(TOOL_HINTS[tool])
        self.refresh_controls()

    def toggle_layers(self):
        self.canvas.set_visibility(dancers=self.dancer_layer.get(), floor=self.floor_layer.get())

    def fit(self):
        if self.frame:
            self.canvas.fit()

    def delete_selected(self):
        if self.frame:
            self.canvas.delete_selection()

    def change_dancer(self, **fields):
        frame = deepcopy(self.frame)
        frame["dancers"][self.selection[1]].update(fields)
        self.apply({self.index: frame})

    def contact_changed(self, _):
        if self.selection[0] != "dancer":
            return
        contact = self.contact_var.get().lower()
        dancer = self.frame["dancers"][self.selection[1]]
        if contact == "visible" and dancer["point"] is None:
            self.set_tool("point")
            self.refresh_controls()
            return
        if contact != "visible":
            self.change_dancer(contact=contact, point=None, support_type="none")

    def support_changed(self, _):
        if self.selection[0] != "dancer":
            return
        name = self.support_var.get()
        support = next(k for k, v in SUPPORT_NAMES.items() if v == name)
        if support == "none":
            self.change_dancer(contact="uncertain", point=None, support_type="none")
        else:
            self.change_dancer(support_type=support)

    def set_id(self):
        from tkinter.simpledialog import askinteger

        kind, index = self.selection
        if kind != "dancer":
            return
        dancers = self.frame["dancers"]
        original = dancers[index]["id"]
        ident = askinteger("Dancer ID", "Enter a positive dancer ID.", initialvalue=original,
                           minvalue=1, parent=self.root)
        if ident is None or ident == original:
            return
        if any(row["id"] == ident for row in dancers):
            self.show_error("Duplicate dancer ID", f"Dancer {ident} already exists on this image.")
            return
        self.change_dancer(id=ident)

    def guide_changed(self, _):
        label = next(k for k, v in FLOOR_NAMES.items() if v == self.guide_kind.get())
        self.canvas.set_floor_label(label)
        kind, index = self.selection
        if kind == "floor" and self.tool != "floor":
            frame = deepcopy(self.frame)
            frame["floor_guides"][index]["label"] = label
            self.apply({self.index: frame})

    def split_changed(self, _):
        if not self.frame:
            return
        split = next(k for k, v in SPLIT_NAMES.items() if v == self.frame_split.get())
        self.canvas.sample_split = split
        frame = deepcopy(self.frame)
        frame["default_split"] = split
        for row in frame["dancers"]:
            row["sample_split"] = split
        self.apply({self.index: frame})
        self.set_status(f"Frame split set to {self.frame_split.get().lower()}.")

    def copy_floor(self):
        if not self.frame or not self.frame["floor_guides"]:
            return
        current = self.frame
        changes = {i: {**frame, "floor_guides": deepcopy(current["floor_guides"])}
                   for i, frame in enumerate(self.project.frames)
                   if not frame["floor_guides"] and (frame["width"], frame["height"]) ==
                   (current["width"], current["height"])}
        if not changes:
            self.set_status("No empty frames with matching dimensions.")
            return
        if messagebox.askyesno("Copy fixed-camera guides",
                               f"Copy these guides to {len(changes):,} empty frames with matching dimensions? "
                               "Use this only when the camera and mat stay fixed.",
                               parent=self.root):
            self.apply(changes)
            self.set_status(f"Copied floor guides to {len(changes):,} frames.")

    def apply(self, changes, update_canvas=True):
        changes = {index: frame for index, frame in changes.items() if frame != self.project.frames[index]}
        if not changes:
            return
        before = {index: self.project.frames[index] for index in changes}
        self.history.append((before, changes))
        self.redo_stack.clear()
        self.write_frames(changes, update_canvas)

    def write_frames(self, changes, update_canvas=True):
        for index, frame in changes.items():
            self.project.frames[index] = frame
            self.refresh_queue_row(index)
        if self.index in changes:
            if update_canvas:
                self.canvas.update_annotations(self.frame["dancers"], self.frame["floor_guides"])
                self.canvas.select(*self.selection)
                self.set_split_for_frame()
            self.refresh_shapes()
        self.refresh_progress()
        self.refresh_controls()
        self.schedule_save()

    def set_split_for_frame(self):
        frame = self.frame
        default = frame.get("default_split", "train")
        splits = {row.get("sample_split", default) for row in frame["dancers"]}
        split = next(iter(splits)) if len(splits) == 1 else default
        self.frame_split.set(SPLIT_NAMES[split] if len(splits) < 2 else "Mixed")
        self.canvas.sample_split = split

    def undo(self):
        self.restore_edit(self.history, self.redo_stack, 0, "Undid the last edit.")

    def redo(self):
        self.restore_edit(self.redo_stack, self.history, 1, "Restored the last edit.")

    def restore_edit(self, source, target, direction, status):
        if not source:
            return
        self.canvas.cancel()
        edit = source.pop()
        target.append(edit)
        changes = edit[direction]
        self.selection = (None, None)
        self.write_frames(changes)
        if self.index not in changes:
            self.show_frame(next(iter(changes)))
        self.set_status(status)

    def schedule_save(self):
        self.dirty = True
        self.save_status.configure(text="Saving draft…", fg=MUTED)
        if self.autosave_job:
            self.root.after_cancel(self.autosave_job)
        self.autosave_job = self.root.after(350, self.flush_save)

    def flush_save(self):
        if self.autosave_job:
            self.root.after_cancel(self.autosave_job)
            self.autosave_job = None
        if not self.dirty or not self.project:
            return True
        try:
            target = self.draft_path(self.project.image_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            save_project(self.project, target)
        except (OSError, ValueError) as error:
            logging.exception("Failed to save local draft")
            self.save_status.configure(text="Draft save failed", fg=DANGER)
            self.set_status(f"Could not save the local draft. {error}")
            return False
        self.dirty = False
        self.save_status.configure(text="Draft saved locally", fg=ACCENT)
        return True

    def leave_project(self):
        return self.flush_save() or messagebox.askyesno(
            "Leave unsaved draft?", "Local recovery could not save the latest changes. Leave this project?",
            parent=self.root,
        )

    def save_as(self):
        if not self.project:
            return
        filename = filedialog.asksaveasfilename(
            title="Save a shareable project", defaultextension=".json",
            initialfile=f"{self.project.image_dir.name}.floor-contact.json",
            filetypes=[("Labeling project", "*.json")], parent=self.root,
        )
        if filename:
            try:
                save_project(self.project, Path(filename))
                self.set_status(f"Project saved as {Path(filename).name}.")
            except (OSError, ValueError) as error:
                self.show_error("Could not save the project", error)

    def import_annotations(self):
        if not self.project:
            messagebox.showinfo("Open an image folder", "Choose the images before importing their annotations.",
                                parent=self.root)
            return
        filename = filedialog.askopenfilename(
            title="Import annotations for the open images",
            filetypes=[("CVAT annotations", "*.xml *.zip"), ("XML", "*.xml"), ("ZIP", "*.zip")], parent=self.root,
        )
        if filename:
            try:
                imported = import_xml(self.project, Path(filename))
                changes = {index: frame for index, frame in enumerate(imported.frames)
                           if frame != self.project.frames[index]}
                self.selection = (None, None)
                self.apply(changes)
                self.set_status(f"Imported CVAT annotations for {len(changes):,} changed images.")
            except (OSError, ValueError) as error:
                self.show_error("Could not import the CVAT annotations", error)

    def export(self):
        if not self.project:
            return
        if not any(frame["dancers"] or frame["floor_guides"] for frame in self.project.frames):
            messagebox.showinfo("No annotations yet", "Add at least one annotation before exporting.", parent=self.root)
            return
        filename = filedialog.asksaveasfilename(
            title="Export the folder's annotations", defaultextension=".xml", initialfile="annotations.xml",
            filetypes=[("CVAT for images XML", "*.xml")], parent=self.root,
        )
        if filename:
            try:
                export_xml(self.project, Path(filename))
                self.set_status(f"Exported {len(self.project.frames):,} images to {Path(filename).name}.")
            except (OSError, ValueError) as error:
                self.show_error("Could not export the annotations", error)

    def close(self):
        if self.leave_project():
            self.root.destroy()


def dark_titlebar(root):
    if sys.platform != "win32":
        return
    root.update_idletasks()
    handle = ctypes.windll.user32.GetParent(root.winfo_id())
    value = ctypes.c_int(1)
    if ctypes.windll.dwmapi.DwmSetWindowAttribute(handle, 20, ctypes.byref(value), ctypes.sizeof(value)):
        ctypes.windll.dwmapi.DwmSetWindowAttribute(handle, 19, ctypes.byref(value), ctypes.sizeof(value))


def main():
    try:
        APP_DATA.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=APP_DATA / "app.log", level=logging.ERROR,
                            format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    except OSError:
        logging.basicConfig(level=logging.ERROR)
    root = tk.Tk()
    try:
        root.iconbitmap(default=str(Path(__file__).with_name("floor_contact.ico")))
    except tk.TclError:
        pass
    Labeler(root)
    try:
        dark_titlebar(root)
    except (AttributeError, OSError):
        pass
    root.mainloop()
