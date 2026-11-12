"""Image-space drawing and editing for the local dance annotator."""

from copy import deepcopy
from math import hypot
import tkinter as tk

from PIL import Image, ImageTk


FLOOR_COLORS = {
    "mat_boundary": "#62e2c0",
    "mat_rear_estimate": "#f1c06e",
    "mat_grid": "#83a9e0",
}
BOX_COLOR = "#8af0cc"
POINT_COLOR = "#e0adff"
ACTIVE = "#fff1ad"
CANVAS_BG = "#111723"
HANDLE_RADIUS = 5


def segment_distance(point, first, second):
    x, y = point
    x1, y1 = first
    dx, dy = second[0] - x1, second[1] - y1
    if dx == dy == 0:
        return hypot(x - x1, y - y1)
    fraction = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
    return hypot(x - x1 - dx * fraction, y - y1 - dy * fraction)


class AnnotationCanvas(tk.Canvas):
    """Paint source pixels and edit a private copy of the frame annotations."""

    def __init__(self, parent, on_change, on_select, on_status=None, **options):
        super().__init__(parent, background=CANVAS_BG, highlightthickness=0, **options)
        self.on_change = on_change
        self.on_select = on_select
        self.on_status = on_status or (lambda message: None)
        self.tool = "select"
        self.floor_label = "mat_boundary"
        self.sample_split = "train"
        self.selection = (None, None)
        self.dancers = []
        self.floor_guides = []
        self._image = None
        self._photo = None
        self._photo_key = None
        self._scale = 1.0
        self._origin = (0.0, 0.0)
        self._fit_view = True
        self._redraw_id = None
        self._gesture = None
        self._vertices = []
        self._hover = None
        self._dancers_visible = True
        self._floor_visible = True
        self.bind("<Configure>", self._on_configure)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", self._double_click)
        self.bind("<ButtonPress-2>", self._start_pan)
        self.bind("<B2-Motion>", self._move_pan)
        self.bind("<ButtonRelease-2>", self._end_pan)
        self.bind("<ButtonPress-3>", self._right_click)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", self._leave)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda event: self.zoom(1.2, (event.x, event.y)))
        self.bind("<Button-5>", lambda event: self.zoom(1 / 1.2, (event.x, event.y)))
        self._queue_redraw()

    @property
    def image_size(self):
        return self._image.size if self._image is not None else None

    def set_frame(self, image_path, dancers, floor_guides):
        if image_path is None:
            image = None
        else:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        if self._image is not None:
            self._image.close()
        self._image = image
        self._photo = self._photo_key = None
        self._gesture = None
        self._vertices.clear()
        self._hover = None
        self.selection = (None, None)
        self.update_annotations(dancers, floor_guides)
        self.fit()

    def update_annotations(self, dancers, floor_guides):
        self.dancers = deepcopy(dancers)
        self.floor_guides = deepcopy(floor_guides)
        self._gesture = None
        self.select(*self.selection)
        self._queue_redraw()

    def set_visibility(self, dancers=True, floor=True):
        self._dancers_visible = dancers
        self._floor_visible = floor
        self._queue_redraw()

    def set_tool(self, tool):
        if tool not in {"select", "box", "point", "floor", "pan"}:
            raise ValueError(f"Unknown canvas tool: {tool}")
        if tool != self.tool:
            self.cancel()
        self.tool = tool
        cursor = {"select": "arrow", "box": "crosshair", "point": "crosshair", "floor": "crosshair", "pan": "fleur"}
        self.configure(cursor=cursor[tool])

    def set_floor_label(self, label):
        if label not in FLOOR_COLORS:
            raise ValueError(f"Unknown floor guide: {label}")
        if label != self.floor_label and self._vertices:
            self.cancel()
        self.floor_label = label

    def select(self, kind, index):
        items = {"dancer": self.dancers, "floor": self.floor_guides}.get(kind, ())
        if index is None or not 0 <= index < len(items):
            kind, index = None, None
        self.selection = (kind, index)
        self._queue_redraw()

    def delete_selection(self):
        kind, index = self.selection
        if kind is None:
            return False
        self.cancel()
        items = self.dancers if kind == "dancer" else self.floor_guides
        del items[index]
        self._commit((None, None))
        return True

    def cancel(self):
        gesture = self._gesture
        if gesture is not None and gesture["kind"] in {"box_move", "box_corner", "point_move", "floor_vertex"}:
            items = self.floor_guides if gesture["kind"] == "floor_vertex" else self.dancers
            items[gesture["index"]] = gesture["original"]
        had_edit = bool(self._vertices) or (gesture is not None and gesture["kind"] != "pan")
        self._gesture = None
        self._vertices.clear()
        self._hover = None
        self._queue_redraw()
        return had_edit

    def finish_floor(self):
        if len(self._vertices) < 2:
            if self._vertices:
                self.on_status("Add another vertex to finish the floor guide.")
            return False
        self.floor_guides.append({"label": self.floor_label, "points": self._vertices.copy()})
        self._vertices.clear()
        self._hover = None
        self._commit(("floor", len(self.floor_guides) - 1))
        return True

    def _user_select(self, kind, index):
        if (kind, index) != self.selection:
            self.select(kind, index)
            self.on_select(*self.selection)

    def _commit(self, selection=None):
        if selection is not None:
            self.select(*selection)
        self._queue_redraw()
        self.on_change(deepcopy(self.dancers), deepcopy(self.floor_guides))
        if selection is not None:
            self.select(*selection)
            self.on_select(*self.selection)

    def _fit_scale(self):
        width, height = self.image_size
        view_width, view_height = self.winfo_width(), self.winfo_height()
        return min(max(1, view_width - 32) / width, max(1, view_height - 32) / height)

    def fit(self):
        self._fit_view = True
        if self._image is not None:
            self._scale = self._fit_scale()
            width, height = self.image_size
            self._origin = (
                (self.winfo_width() - width * self._scale) / 2,
                (self.winfo_height() - height * self._scale) / 2,
            )
        self._queue_redraw()

    def zoom(self, factor, anchor=None):
        if self._image is None:
            return
        if anchor is None:
            anchor = (self.winfo_width() / 2, self.winfo_height() / 2)
        fit_scale = self._fit_scale()
        new_scale = max(fit_scale / 2, min(max(fit_scale, 6.0), self._scale * factor))
        if new_scale == self._scale:
            return
        x, y = self._to_image(*anchor)
        self._fit_view = False
        self._scale = new_scale
        self._origin = (anchor[0] - x * new_scale, anchor[1] - y * new_scale)
        self._queue_redraw()

    def _on_configure(self, event):
        if self._fit_view:
            self.fit()
        else:
            self._queue_redraw()

    def _wheel(self, event):
        if event.delta:
            self.zoom(1.2 ** (event.delta / 120), (event.x, event.y))
        return "break"

    def _to_screen(self, x, y):
        return self._origin[0] + x * self._scale, self._origin[1] + y * self._scale

    def _to_image(self, x, y, clamp=False):
        x = (x - self._origin[0]) / self._scale
        y = (y - self._origin[1]) / self._scale
        if clamp:
            width, height = self.image_size
            x = max(0.0, min(width, x))
            y = max(0.0, min(height, y))
        return x, y

    def _inside(self, point):
        width, height = self.image_size
        return 0 <= point[0] < width and 0 <= point[1] < height

    def _queue_redraw(self):
        if self._redraw_id is None:
            self._redraw_id = self.after_idle(self._redraw)

    def _redraw(self):
        self._redraw_id = None
        self.delete("all")
        if self._image is None:
            return
        self._draw_image()
        if self._floor_visible:
            for index, guide in enumerate(self.floor_guides):
                self._draw_guide(guide, self.selection == ("floor", index))
        if self._dancers_visible:
            for index, dancer in enumerate(self.dancers):
                if self.selection != ("dancer", index):
                    self._draw_dancer(dancer, False)
            kind, index = self.selection
            if kind == "dancer":
                self._draw_dancer(self.dancers[index], True)
        if self._vertices:
            self._draw_floor_draft()
        gesture = self._gesture
        if gesture is not None and gesture["kind"] == "box_new":
            x1, y1 = self._to_screen(*gesture["start"])
            x2, y2 = self._to_screen(*gesture["current"])
            self.create_rectangle(x1, y1, x2, y2, outline=ACTIVE, width=2, dash=(5, 3))

    def _draw_image(self):
        width, height = self.image_size
        x0, y0 = self._origin
        vw, vh = self.winfo_width(), self.winfo_height()
        left = max(0, round(x0))
        top = max(0, round(y0))
        right = min(vw, round(x0 + width * self._scale))
        bottom = min(vh, round(y0 + height * self._scale))
        if right <= left or bottom <= top:
            return
        key = (self._origin, self._scale, vw, vh)
        if key != self._photo_key:
            x1, y1 = self._to_image(left, top, clamp=True)
            x2, y2 = self._to_image(right, bottom, clamp=True)
            cropped = self._image.resize(
                (right - left, bottom - top),
                resample=Image.Resampling.BILINEAR,
                box=(x1, y1, x2, y2),
            )
            self._photo = ImageTk.PhotoImage(cropped, master=self)
            self._photo_key = key
        self.create_image(left, top, image=self._photo, anchor="nw")

    def _draw_guide(self, guide, selected):
        points = [self._to_screen(*point) for point in guide["points"]]
        if len(points) < 2:
            return
        color = ACTIVE if selected else FLOOR_COLORS[guide["label"]]
        xy = [coordinate for point in points for coordinate in point]
        options = {"dash": (7, 4)} if guide["label"] == "mat_rear_estimate" else {}
        self.create_line(*xy, fill="#152133", width=5 if selected else 3, **options)
        self.create_line(*xy, fill=color, width=3 if selected else 1, **options)
        if selected and self.tool == "select":
            for x, y in points:
                self._handle(x, y, color)

    def _draw_dancer(self, dancer, selected):
        x1, y1, x2, y2 = dancer["box"]
        a, b = self._to_screen(x1, y1)
        c, d = self._to_screen(x2, y2)
        color = ACTIVE if selected else BOX_COLOR
        self.create_rectangle(a, b, c, d, outline="#122134", width=4)
        self.create_rectangle(a, b, c, d, outline=color, width=2 if selected else 1)
        label = self.create_text(a + 5, b + 3, anchor="nw", fill=color, text=f'{dancer["id"]}', font=("Segoe UI", 9, "bold"))
        left, top, right, bottom = self.bbox(label)
        background = self.create_rectangle(left - 4, top - 1, right + 4, bottom + 1, fill="#182435", outline="")
        self.tag_lower(background, label)
        if selected and self.tool == "select":
            for x, y in ((a, b), (c, b), (c, d), (a, d)):
                self._handle(x, y, color)
        if dancer["point"] is not None:
            px, py = self._to_screen(*dancer["point"])
            radius = 7 if selected else 4
            self.create_oval(px - radius, py - radius, px + radius, py + radius, fill=POINT_COLOR, outline="#22253b", width=2)
            if selected:
                self.create_oval(px - 2, py - 2, px + 2, py + 2, fill="#302644", outline="")

    def _handle(self, x, y, color):
        r = HANDLE_RADIUS
        self.create_rectangle(x - r, y - r, x + r, y + r, fill="#111723", outline=color, width=2)

    def _draw_floor_draft(self):
        points = [self._to_screen(*point) for point in self._vertices]
        if self._hover is not None:
            points.append(self._to_screen(*self._hover))
        if len(points) > 1:
            xy = [coordinate for point in points for coordinate in point]
            self.create_line(*xy, fill=ACTIVE, width=2, dash=(5, 3))
        for x, y in points[:len(self._vertices)]:
            self._handle(x, y, ACTIVE)

    def _motion(self, event):
        if self._image is not None and self.tool == "floor" and self._vertices:
            self._hover = self._to_image(event.x, event.y, clamp=True)
            self._queue_redraw()

    def _leave(self, event):
        if self._hover is not None:
            self._hover = None
            self._queue_redraw()

    def _near(self, image_point, screen_point, distance=8):
        x, y = self._to_screen(*image_point)
        return hypot(x - screen_point[0], y - screen_point[1]) <= distance

    def _dancer_at(self, point, screen_point):
        if not self._dancers_visible:
            return None
        for index in reversed(range(len(self.dancers))):
            support = self.dancers[index]["point"]
            if support is not None and self._near(support, screen_point):
                return index
        candidates = []
        for index, dancer in enumerate(self.dancers):
            x1, y1, x2, y2 = dancer["box"]
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                candidates.append(((x2 - x1) * (y2 - y1), index))
        return min(candidates)[1] if candidates else None

    def _floor_at(self, screen_point):
        if not self._floor_visible:
            return None
        best_distance = 7.0
        closest = None
        for index, guide in enumerate(self.floor_guides):
            vertices = [self._to_screen(*point) for point in guide["points"]]
            for first, second in zip(vertices, vertices[1:]):
                distance = segment_distance(screen_point, first, second)
                if distance < best_distance:
                    best_distance, closest = distance, index
        return closest

    def _press(self, event):
        if self._image is None:
            return
        self.focus_set()
        if self.tool == "pan":
            self._start_pan(event)
            return
        screen_point = event.x, event.y
        point = self._to_image(*screen_point)
        if self.tool == "select":
            self._select_press(point, screen_point)
        elif self._inside(point):
            if self.tool == "box":
                self._gesture = {"kind": "box_new", "start": point, "current": point}
                self._queue_redraw()
            elif self.tool == "point":
                self._place_point(point, screen_point)
            elif self.tool == "floor":
                self._add_vertex(point, screen_point)

    def _double_click(self, event):
        if self.tool == "floor" and self._image is not None:
            screen = (event.x, event.y)
            point = self._to_image(*screen)
            if self._inside(point):
                self._add_vertex(point, screen)
            self.finish_floor()
            return "break"
        # Tk dispatches the double-click binding in place of the second press.
        self._press(event)

    def _right_click(self, event):
        if self.tool == "floor" and self._vertices:
            self._vertices.pop()
            self._queue_redraw()
            return "break"

    def _add_vertex(self, point, screen_point):
        if self._vertices and self._near(self._vertices[-1], screen_point, distance=3):
            return
        self._vertices.append(list(point))
        self._hover = point
        self._queue_redraw()

    def _place_point(self, point, screen_point):
        kind, index = self.selection
        if kind != "dancer":
            index = self._dancer_at(point, screen_point)
        if index is None:
            self.on_status("Select a dancer or click inside their box to place a support point.")
            return
        dancer = self.dancers[index]
        dancer["point"] = list(point)
        dancer["contact"] = "visible"
        if dancer["support_type"] in (None, "none"):
            dancer["support_type"] = "feet"
        self._commit(("dancer", index))

    def _edit(self, kind, index, point, **details):
        items = self.floor_guides if kind == "floor_vertex" else self.dancers
        self._gesture = {"kind": kind, "index": index, "start": point, "original": deepcopy(items[index]), **details}

    def _select_press(self, point, screen_point):
        kind, index = self.selection
        if kind == "dancer" and self._dancers_visible:
            dancer = self.dancers[index]
            if dancer["point"] is not None and self._near(dancer["point"], screen_point):
                self._edit("point_move", index, point)
                return
            x1, y1, x2, y2 = dancer["box"]
            for x, y, left, top in ((x1, y1, True, True), (x2, y1, False, True), (x2, y2, False, False), (x1, y2, True, False)):
                if self._near((x, y), screen_point):
                    self._edit("box_corner", index, point, left=left, top=top)
                    return
        elif kind == "floor" and self._floor_visible:
            for vertex, position in enumerate(self.floor_guides[index]["points"]):
                if self._near(position, screen_point):
                    self._edit("floor_vertex", index, point, vertex=vertex)
                    return

        floor_index = self._floor_at(screen_point)
        dancer_index = self._dancer_at(point, screen_point)
        if dancer_index is not None:
            self._user_select("dancer", dancer_index)
            support = self.dancers[dancer_index]["point"]
            if support is not None and self._near(support, screen_point):
                self._edit("point_move", dancer_index, point)
            else:
                self._edit("box_move", dancer_index, point)
        elif floor_index is not None:
            self._user_select("floor", floor_index)
        else:
            self._user_select(None, None)

    def _drag(self, event):
        gesture = self._gesture
        if gesture is None:
            return
        kind = gesture["kind"]
        if kind == "pan":
            self._move_pan(event)
            return
        point = self._to_image(event.x, event.y, clamp=True)
        if kind == "box_new":
            gesture["current"] = point
        else:
            self._apply_drag(gesture, point)
        self._queue_redraw()

    def _apply_drag(self, gesture, point):
        kind, index = gesture["kind"], gesture["index"]
        original = gesture["original"]
        dx, dy = point[0] - gesture["start"][0], point[1] - gesture["start"][1]
        width, height = self.image_size
        if kind in {"floor_vertex", "point_move"}:
            if kind == "floor_vertex":
                vertex = gesture["vertex"]
                initial = original["points"][vertex]
                target = self.floor_guides[index]["points"]
                key = vertex
            else:
                initial = original["point"]
                target = self.dancers[index]
                key = "point"
            # A point must stay strictly inside the image, including after XML serialization.
            target[key] = [
                max(0, min(width - 0.001, initial[0] + dx)),
                max(0, min(height - 0.001, initial[1] + dy)),
            ]
        else:
            x1, y1, x2, y2 = original["box"]
            if kind == "box_move":
                dx = max(-x1, min(width - x2, dx))
                dy = max(-y1, min(height - y2, dy))
                box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            else:
                # Keep the opposite corner fixed and retain positive area.
                min_width, min_height = min(1, x2 - x1), min(1, y2 - y1)
                if gesture["left"]:
                    x1 = max(0, min(x2 - min_width, x1 + dx))
                else:
                    x2 = min(width, max(x1 + min_width, x2 + dx))
                if gesture["top"]:
                    y1 = max(0, min(y2 - min_height, y1 + dy))
                else:
                    y2 = min(height, max(y1 + min_height, y2 + dy))
                box = [x1, y1, x2, y2]
            self.dancers[index]["box"] = box

    def _release(self, event):
        gesture = self._gesture
        if gesture is None:
            return
        if gesture["kind"] == "pan":
            self._end_pan(event)
            return
        self._drag(event)
        self._gesture = None
        if gesture["kind"] == "box_new":
            self._add_box(gesture)
            return
        index = gesture["index"]
        items = self.floor_guides if gesture["kind"] == "floor_vertex" else self.dancers
        if items[index] != gesture["original"]:
            self._commit()

    def _add_box(self, gesture):
        x1, x2 = sorted((gesture["start"][0], gesture["current"][0]))
        y1, y2 = sorted((gesture["start"][1], gesture["current"][1]))
        if x2 - x1 < max(1, 3 / self._scale) or y2 - y1 < max(1, 3 / self._scale):
            return
        ids = {dancer["id"] for dancer in self.dancers}
        ident = next(number for number in range(1, len(ids) + 2) if number not in ids)
        self.dancers.append({
            "id": ident, "box": [x1, y1, x2, y2], "point": None, "contact": "uncertain",
            "support_type": "none", "sample_split": self.sample_split,
        })
        self._commit(("dancer", len(self.dancers) - 1))

    def _start_pan(self, event):
        if self._image is None:
            return
        if self._gesture is not None:
            return
        self._gesture = {"kind": "pan", "start": (event.x, event.y), "origin": self._origin}

    def _move_pan(self, event):
        gesture = self._gesture
        if gesture is not None and gesture["kind"] == "pan":
            self._fit_view = False
            x, y = gesture["start"]
            left, top = gesture["origin"]
            self._origin = (left + event.x - x, top + event.y - y)
            self._queue_redraw()

    def _end_pan(self, event):
        if self._gesture is not None and self._gesture["kind"] == "pan":
            self._gesture = None
