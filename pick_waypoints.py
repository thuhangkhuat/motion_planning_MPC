"""
pick_waypoints.py — Pick target waypoints with the mouse on the map.

    python pick_waypoints.py            # scenario from config / env SCENARIO
    python pick_waypoints.py 3          # scenario 3

Controls:
    Left click (empty space)    append a point at the END
    Left click + drag (point)   move the point
    Shift + left click          INSERT a point into the nearest segment
    Right click (point)         delete the point
    z  undo         c  clear all     p  toggle spline preview
    s  save         Enter  save and quit
    (Zoom/Pan via the toolbar: while zoom/pan is active, clicks add no points.)

Saved to scenarios/target_scen{N}.json — main.py reads this file automatically.
Marked x = piece too close to / through an obstacle
(clearance < TAR_RADIUS + SAFETY_MARGIN).
"""

import os
import sys

if len(sys.argv) > 1 and sys.argv[1].isdigit():
    os.environ["SCENARIO"] = sys.argv[1]      # must be set BEFORE importing config

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle

from config import (SCENARIO, XLIM, YLIM, STARTS, RECTANGLE_OBSTACLES, OBSTACLES,
                    TAR_RADIUS, SAFETY_MARGIN, TAR_MAX_SPEED, TAR_SPLINE_DS)
from target_manual import (load_waypoints, save_waypoints,
                           catmull_rom, check_path)

MARGIN = TAR_RADIUS + SAFETY_MARGIN
PICK_PX = 10          # hit radius (pixels) for selecting a point


class Picker:
    def __init__(self):
        wps, speeds, src = load_waypoints()
        self.pts = [list(map(float, p)) for p in wps]
        self.speeds = speeds
        self.history = []
        self.drag = None
        self.show_spline = True
        self.dirty = False

        w = 11
        h = max(5, w * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0]))
        self.fig, self.ax = plt.subplots(figsize=(w, min(h, 11)))
        ax = self.ax
        patches = [Rectangle((r[0], r[1]), r[2], r[3]) for r in RECTANGLE_OBSTACLES]
        patches += [Circle((c[0], c[1]), c[2]) for c in (OBSTACLES if len(OBSTACLES) else [])]
        ax.add_collection(PatchCollection(patches, facecolor="0.6", edgecolor="k", lw=0.5))
        ax.scatter(STARTS[:, 0], STARTS[:, 1], marker="^", s=90, c="tab:blue",
                   edgecolors="k", zorder=4, label="UAV start")
        ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect("equal")
        ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

        (self.l_poly,) = ax.plot([], [], "--", c="tab:red", lw=1, alpha=0.6)
        (self.l_spl,) = ax.plot([], [], "-", c="tab:red", lw=2, label="Target path")
        self.sc = ax.scatter([], [], s=60, c="tab:red", edgecolors="darkred", zorder=5)
        self.sc_bad = ax.scatter([], [], s=120, marker="x", c="k", zorder=6,
                                 label="Collision")
        self.labels = []
        ax.legend(loc="upper right", fontsize=8)

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("key_press_event", self.on_key)
        print(__doc__)
        print(f"[PICK] Loaded {len(self.pts)} points from {src}")
        self.redraw()

    # ---------- helpers ----------
    def _toolbar_busy(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return tb is not None and getattr(tb, "mode", "") not in ("", None)

    def _hit(self, event):
        """Index of the point under the cursor (in pixels), or None."""
        if not self.pts:
            return None
        xy = self.ax.transData.transform(np.asarray(self.pts))
        d = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
        i = int(np.argmin(d))
        return i if d[i] < PICK_PX else None

    def _snapshot(self):
        self.history.append([p[:] for p in self.pts])
        self.dirty = True

    def _insert_index(self, p):
        """Insert position: after the start of the segment closest to p."""
        P = np.asarray(self.pts)
        best, idx = np.inf, len(self.pts)
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
            d = np.linalg.norm(a + t * (b - a) - p)
            if d < best:
                best, idx = d, i + 1
        return idx

    # ---------- events ----------
    def on_press(self, e):
        if e.inaxes is not self.ax or self._toolbar_busy():
            return
        i = self._hit(e)
        p = [float(e.xdata), float(e.ydata)]
        if e.button == 1:
            if i is not None:
                self._snapshot(); self.drag = i
            elif e.key == "shift" and len(self.pts) >= 2:
                self._snapshot(); self.pts.insert(self._insert_index(np.array(p)), p)
            else:
                self._snapshot(); self.pts.append(p)
        elif e.button == 3 and i is not None:
            self._snapshot(); self.pts.pop(i)
        self.redraw()

    def on_move(self, e):
        if self.drag is None or e.inaxes is not self.ax:
            return
        self.pts[self.drag] = [float(e.xdata), float(e.ydata)]
        self.redraw(fast=True)

    def on_release(self, e):
        if self.drag is not None:
            self.drag = None
            self.redraw()

    def on_key(self, e):
        if e.key == "z" and self.history:
            self.pts = self.history.pop()
        elif e.key == "c":
            self._snapshot(); self.pts = []
        elif e.key == "p":
            self.show_spline = not self.show_spline
        elif e.key == "s":
            self.save()
        elif e.key == "enter":
            self.save(); plt.close(self.fig); return
        self.redraw()

    # ---------- drawing ----------
    def redraw(self, fast=False):
        P = np.asarray(self.pts) if self.pts else np.empty((0, 2))
        self.sc.set_offsets(P)
        self.l_poly.set_data(P[:, 0], P[:, 1])
        for t in self.labels:
            t.remove()
        self.labels = [self.ax.annotate(f"W{i}", p, xytext=(6, 6), textcoords="offset points",
                                        fontsize=9, color="darkred") for i, p in enumerate(P)]
        path, knots = P, np.arange(len(P))
        if self.show_spline and len(P) >= 3:
            path, knots = catmull_rom(P, max(TAR_SPLINE_DS, 1e-3))
        self.l_spl.set_data(path[:, 0], path[:, 1]) if len(path) else self.l_spl.set_data([], [])

        bad = [] if fast or len(path) < 2 else check_path(path, MARGIN)
        self.sc_bad.set_offsets(np.array([b[1] for b in bad]) if bad else np.empty((0, 2)))

        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
        bad_seg = sorted({int(np.searchsorted(knots, b[0] + 1) - 1) for b in bad})
        status = ("collision on " + ", ".join(f"W{k}→W{k+1}" for k in bad_seg)) if bad else "OK"
        self.ax.set_title(f"Scenario {SCENARIO} — {len(P)} points, length {length:.1f} m, "
                          f"~{length / TAR_MAX_SPEED:.0f} s @ {TAR_MAX_SPEED} m/s — {status}"
                          + ("  (unsaved)" if self.dirty else ""), fontsize=10)
        self.fig.canvas.draw_idle()

    def save(self):
        if len(self.pts) < 2:
            print("[PICK] At least 2 points are required, not saved.")
            return
        speeds = self.speeds
        if speeds is not None and len(speeds) != len(self.pts) - 1:
            print("[PICK] Point count changed -> dropping old 'speeds' (using TAR_MAX_SPEED).")
            speeds = None
        path = save_waypoints(self.pts, speeds)
        self.dirty = False
        print(f"[PICK] Saved {len(self.pts)} points to {path}")
        self.redraw()


if __name__ == "__main__":
    Picker()
    plt.show()