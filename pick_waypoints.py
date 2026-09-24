"""
pick_waypoints.py — Pick target waypoints and UAV start positions on the map.

    python pick_waypoints.py            # scenario from env SCENARIO / config
    python pick_waypoints.py 6          # scenarios/scen6.yaml
    python pick_waypoints.py big1000    # scenarios/big1000.yaml

Two layers, switch with the keyboard:
    t   edit TARGET waypoints (red)        u   edit UAV STARTS (blue)

Mouse (acts on the active layer):
    Left click (empty space)    append a point at the END
    Left click + drag (point)   move the point
    Shift + left click          target: INSERT into the nearest segment
    Right click (point)         delete the point

Keys:
    z  undo     c  clear active layer     p  toggle spline preview
    r  regenerate the random map around the current points (generated maps only)
    s  save     Enter  save and quit
    (Zoom/Pan via the toolbar: while zoom/pan is active, clicks add no points.)

Saved to scenarios/<name>.picks.json, which overrides the YAML values.
Marked x = target path too close to / through an obstacle
(clearance < TAR_RADIUS + SAFETY_MARGIN). Starts inside an obstacle are
drawn with a red edge.
"""

import os
import sys

if len(sys.argv) > 1:
    os.environ["SCENARIO"] = sys.argv[1]      # must be set BEFORE importing config

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle

import config
from config import (SCENARIO, SCENARIO_NAME, XLIM, YLIM, STARTS, START_ALTITUDE,
                    TAR_RADIUS, SAFETY_MARGIN, TAR_MAX_SPEED, TAR_SPLINE_DS, ROBOT_RADIUS,
                    TAR_SMOOTH_ENABLE)
from target_manual import load_waypoints, save_picks, catmull_rom, check_path, clearance

MARGIN = TAR_RADIUS + SAFETY_MARGIN
PICK_PX = 10          # hit radius (pixels) for selecting a point


class Picker:
    def __init__(self):
        wps, speeds, src = load_waypoints()
        self.layers = {"target": [list(map(float, p)) for p in wps],
                       "starts": [list(map(float, p)) for p in STARTS]}
        self.active = "target"
        self.speeds = speeds
        self.history = []
        self.drag = None
        self.show_spline = True
        self.dirty = False
        self.rects = list(config.RECTANGLE_OBSTACLES)
        self.circles = np.asarray(config.OBSTACLES).reshape(-1, 3)
        self.raw = config.load_scenario(SCENARIO)

        w = 11
        h = max(5, w * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0]))
        self.fig, self.ax = plt.subplots(figsize=(w, min(h, 11)))
        ax = self.ax
        self.obs_coll = None
        self._draw_obstacles()
        ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect("equal")
        ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

        (self.l_poly,) = ax.plot([], [], "--", c="tab:red", lw=1, alpha=0.6)
        (self.l_spl,) = ax.plot([], [], "-", c="tab:red", lw=2, label="Target path")
        self.sc = ax.scatter([], [], s=60, c="tab:red", edgecolors="darkred", zorder=5)
        self.sc_bad = ax.scatter([], [], s=120, marker="x", c="k", zorder=6, label="Collision")
        self.sc_uav = ax.scatter([], [], marker="^", s=110, c="tab:blue", zorder=6,
                                 label="UAV start")
        self.labels = []
        ax.legend(loc="upper right", fontsize=8)

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("key_press_event", self.on_key)
        print(__doc__)
        print(f"[PICK] Loaded {len(self.layers['target'])} waypoints and "
              f"{len(self.layers['starts'])} starts from {src}")
        self.redraw()

    # ---------- obstacles ----------
    def _draw_obstacles(self):
        if self.obs_coll is not None:
            self.obs_coll.remove()
        patches = [Rectangle((r[0], r[1]), r[2], r[3]) for r in self.rects]
        patches += [Circle((c[0], c[1]), c[2]) for c in self.circles]
        self.obs_coll = PatchCollection(patches, facecolor="0.6", edgecolor="k", lw=0.5)
        self.ax.add_collection(self.obs_coll)

    def regenerate(self):
        """Re-run map_gen with keep-clear zones at the current points."""
        gen = self.raw.get("generate")
        if not gen:
            print("[PICK] This scenario has no `generate:` block.")
            return
        from map_gen import generate_obstacles
        keep = [p[:2] for p in self.layers["starts"]] + [p[:2] for p in self.layers["target"]]
        W = np.asarray(self.layers["target"], float).reshape(-1, 2)
        if TAR_SMOOTH_ENABLE and len(W) >= 3:
            W = catmull_rom(W, max(TAR_SPLINE_DS, 1e-3))[0]
        rects, circles, st = generate_obstacles(
            gen, XLIM, YLIM, keep_clear_points=keep,
            fixed_rects=self.raw["obstacles"].get("rects") or [],
            fixed_circles=self.raw["obstacles"].get("circles") or [],
            target_path=W)
        self.rects = [np.asarray(r, float) for r in rects]
        self.circles = np.asarray(circles, float).reshape(-1, 3)
        self._draw_obstacles()
        print(f"[PICK] Regenerated map: {st['n_rects']} rects, {st['n_circles']} circles")

    # ---------- helpers ----------
    @property
    def pts(self):
        return self.layers[self.active]

    def _toolbar_busy(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return tb is not None and getattr(tb, "mode", "") not in ("", None)

    def _hit(self, event):
        """Index of the active-layer point under the cursor (in pixels), or None."""
        if not self.pts:
            return None
        xy = self.ax.transData.transform(np.asarray(self.pts)[:, :2])
        d = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
        i = int(np.argmin(d))
        return i if d[i] < PICK_PX else None

    def _snapshot(self):
        self.history.append({k: [p[:] for p in v] for k, v in self.layers.items()})
        self.dirty = True

    def _insert_index(self, p):
        """Insert position: after the start of the segment closest to p."""
        P = np.asarray(self.pts)[:, :2]
        best, idx = np.inf, len(P)
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
            d = np.linalg.norm(a + t * (b - a) - p)
            if d < best:
                best, idx = d, i + 1
        return idx

    def _new_point(self, x, y):
        if self.active == "target":
            return [x, y]
        z = self.layers["starts"][-1][2] if self.layers["starts"] else START_ALTITUDE
        return [x, y, z]

    # ---------- events ----------
    def on_press(self, e):
        if e.inaxes is not self.ax or self._toolbar_busy():
            return
        i = self._hit(e)
        p = self._new_point(float(e.xdata), float(e.ydata))
        if e.button == 1:
            if i is not None:
                self._snapshot(); self.drag = i
            elif e.key == "shift" and self.active == "target" and len(self.pts) >= 2:
                self._snapshot(); self.pts.insert(self._insert_index(np.array(p[:2])), p)
            else:
                self._snapshot(); self.pts.append(p)
        elif e.button == 3 and i is not None:
            self._snapshot(); self.pts.pop(i)
        self.redraw()

    def on_move(self, e):
        if self.drag is None or e.inaxes is not self.ax:
            return
        self.pts[self.drag][:2] = [float(e.xdata), float(e.ydata)]
        self.redraw(fast=True)

    def on_release(self, e):
        if self.drag is not None:
            self.drag = None
            self.redraw()

    def on_key(self, e):
        if e.key == "z" and self.history:
            self.layers = self.history.pop()
        elif e.key == "c":
            self._snapshot(); self.layers[self.active] = []
        elif e.key == "p":
            self.show_spline = not self.show_spline
        elif e.key == "t":
            self.active = "target"
        elif e.key == "u":
            self.active = "starts"
        elif e.key == "r":
            self.regenerate()
        elif e.key == "s":
            self.save()
        elif e.key == "enter":
            self.save(); plt.close(self.fig); return
        self.redraw()

    # ---------- drawing ----------
    def redraw(self, fast=False):
        for t in self.labels:
            t.remove()
        self.labels = []

        # target layer
        P = np.asarray(self.layers["target"]).reshape(-1, 2)
        self.sc.set_offsets(P)
        self.l_poly.set_data(P[:, 0], P[:, 1])
        self.labels += [self.ax.annotate(f"W{i}", p, xytext=(6, 6), textcoords="offset points",
                                         fontsize=9, color="darkred") for i, p in enumerate(P)]
        path, knots = P, np.arange(len(P))
        if self.show_spline and len(P) >= 3:
            path, knots = catmull_rom(P, max(TAR_SPLINE_DS, 1e-3))
        self.l_spl.set_data(path[:, 0], path[:, 1])
        bad = [] if fast or len(path) < 2 else check_path(path, MARGIN, rects=self.rects,
                                                           circles=self.circles)
        self.sc_bad.set_offsets(np.array([b[1] for b in bad]) if bad else np.empty((0, 2)))

        # starts layer
        S = np.asarray(self.layers["starts"]).reshape(-1, 3)[:, :2]
        self.sc_uav.set_offsets(S)
        if len(S):
            inside = clearance(S, self.rects, self.circles) < ROBOT_RADIUS
            self.sc_uav.set_edgecolors(np.where(inside[:, None], [[1, 0, 0, 1]], [[0, 0, 0, 1]]))
            self.sc_uav.set_linewidths(np.where(inside, 2.5, 0.8))
        self.labels += [self.ax.annotate(f"U{i}", p, xytext=(6, -12), textcoords="offset points",
                                         fontsize=9, color="navy") for i, p in enumerate(S)]

        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
        bad_seg = sorted({int(np.searchsorted(knots, b[0] + 1) - 1) for b in bad})
        status = ("collision on " + ", ".join(f"W{k}→W{k+1}" for k in bad_seg)) if bad else "OK"
        mode = "TARGET [t]" if self.active == "target" else "UAV STARTS [u]"
        self.ax.set_title(f"{SCENARIO_NAME} — editing {mode} — {len(P)} waypoints, "
                          f"{len(S)} UAVs, path {length:.1f} m "
                          f"(~{length / TAR_MAX_SPEED:.0f} s) — {status}"
                          + ("  (unsaved)" if self.dirty else ""), fontsize=9)
        self.fig.canvas.draw_idle()

    def save(self):
        if len(self.layers["target"]) < 2:
            print("[PICK] At least 2 target waypoints are required, not saved.")
            return
        if len(self.layers["starts"]) < 1:
            print("[PICK] At least 1 UAV start is required, not saved.")
            return
        speeds = self.speeds
        if speeds is not None and len(speeds) != len(self.layers["target"]) - 1:
            print("[PICK] Waypoint count changed -> dropping old 'speeds' (using TAR_MAX_SPEED).")
            speeds = None
        path = save_picks(self.layers["target"], speeds, self.layers["starts"])
        self.dirty = False
        print(f"[PICK] Saved {len(self.layers['target'])} waypoints and "
              f"{len(self.layers['starts'])} starts to {os.path.relpath(path)}")
        self.redraw()


if __name__ == "__main__":
    Picker()
    plt.show()
