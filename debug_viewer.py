"""
debug_viewer.py — Step through a finished run, with the view cropped around the target.

    python debug_viewer.py                          # latest run in runs/
    python debug_viewer.py runs/<dir>               # a given run (or its data.pkl)
    python debug_viewer.py runs/<dir> --start 1200 --window 80 --speed 4

Keys:
    space        play / pause                 [ / ]   playback speed /2, x2 (x1 = real time)
    right / left +-1 step                     up / down  +-10 steps
    pageup/down  +-100 steps                  home / end first / last step
    n / b        next / previous step where an MPC solve failed
    + / -        zoom in / out (half size of the window around the target)
    f            follow the target <-> whole map
    c  corridors     m  MPC predictions     r  planner reference     t  trails
    q / escape   quit
The slider at the bottom jumps to any step.

Everything comes from the run directory (data.pkl + config.json), so the map is
the one the run used even if the scenario YAML changed since. MPC predictions
and UAV modes are only in runs made after debug recording was added.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

# Bypass the input method (IBus + Unikey would swallow letter keys, see pick_waypoints.py)
os.environ["XMODIFIERS"] = ""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.widgets import Slider

# viewer keys that clash with matplotlib's default shortcuts
for _name in ("keymap.fullscreen", "keymap.home", "keymap.back", "keymap.forward",
              "keymap.pan", "keymap.zoom", "keymap.save", "keymap.quit", "keymap.grid",
              "keymap.xscale", "keymap.yscale"):
    plt.rcParams[_name] = []

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
LEADER_COLOR = "#f1c40f"
TRAIL = 150            # steps of trail drawn behind each UAV / the target


# ============================================================
# Loading
# ============================================================
def find_run(arg):
    if arg is None:
        runs = sorted(glob.glob("runs/*/data.pkl"), key=os.path.getmtime)
        if not runs:
            sys.exit("No run found in runs/. Run main.py first or pass a run directory.")
        return os.path.dirname(runs[-1])
    return os.path.dirname(arg) if arg.endswith(".pkl") else arg


def load_run(run_dir):
    with open(os.path.join(run_dir, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else {}
    return data, cfg


def halfspace_polygon(A, b, box):
    """Vertices of {p : A p <= b} clipped to box = (xmin, xmax, ymin, ymax)."""
    x0, x1, y0, y1 = box
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for a, c in zip(np.asarray(A, float).reshape(-1, 2), np.asarray(b, float).reshape(-1)):
        out = []
        for k in range(len(poly)):
            p, q = np.array(poly[k]), np.array(poly[(k + 1) % len(poly)])
            fp, fq = a @ p - c, a @ q - c
            if fp <= 0:
                out.append(tuple(p))
            if fp * fq < 0:
                out.append(tuple(p + (q - p) * fp / (fp - fq)))
        poly = out
        if not poly:
            break
    return np.array(poly) if len(poly) >= 3 else None


# ============================================================
# Viewer
# ============================================================
class Viewer:
    def __init__(self, run_dir, start, window, speed=1.0):
        data, cfg = load_run(run_dir)
        sd, params = cfg.get("scenario_def", {}), cfg.get("params", {})
        self.ids = sorted(k for k in data if isinstance(k, int))
        self.paths = [np.asarray(data[i]["path"]) for i in self.ids]
        self.tar = np.asarray(data[self.ids[0]]["tar_traj"])[:, :2]
        self.T = min(len(self.tar), *[len(p) for p in self.paths])
        self.refs = [data[i].get("traj_refs") for i in self.ids]
        self.corr = [data[i].get("corridors") for i in self.ids]
        self.pred = [data[i].get("mpc_pred") for i in self.ids]
        self.status = [data[i].get("status") for i in self.ids]
        n = len(self.ids)
        ct = np.asarray(data.get("meta", {}).get("compute_times", []))
        self.ct = ct[:self.T * n].reshape(-1, n) if ct.size >= n else None

        self.VR = float(sd.get("viewing_radius", params.get("VIEWING_RADIUS", 1.0)))
        self.R = float(params.get("ROBOT_RADIUS", 0.2))
        self.dt = float(params.get("TIMESTEP", 0.1))
        self.xlim = sd.get("xlim", [self.tar[:, 0].min() - 10, self.tar[:, 0].max() + 10])
        self.ylim = sd.get("ylim", [self.tar[:, 1].min() - 10, self.tar[:, 1].max() + 10])
        self.W = window or 4 * self.VR
        self.fails = np.array(sorted({k for s in self.status if s is not None
                                      for k in np.flatnonzero(s[:self.T, 2] > 0)}), int)

        self.k = int(np.clip(start, 0, self.T - 1))
        # speed = simulated seconds per wall-clock second; playback skips steps when
        # drawing cannot keep up, so x1 is real time whatever the frame rate
        self.follow, self.playing, self.speed = True, False, float(speed)
        self._t_last, self._carry = None, 0.0
        self.show = {"corr": True, "pred": True, "ref": True, "trail": True}

        # ── figure ──
        self.fig = plt.figure(figsize=(13, 8.5))
        self.ax = self.fig.add_axes([0.04, 0.10, 0.62, 0.86])
        self.info = self.fig.add_axes([0.68, 0.10, 0.31, 0.86])
        self.info.axis("off")
        ax = self.ax
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

        # static: obstacles (drawn once) + full target path
        patches = [Rectangle((r[0], r[1]), r[2], r[3]) for r in sd.get("rects", [])]
        patches += [Circle((c[0], c[1]), c[2]) for c in sd.get("circles", [])]
        ax.add_collection(PatchCollection(patches, fc="0.35", ec="k", lw=0.5, zorder=1))
        ax.plot(self.tar[:self.T, 0], self.tar[:self.T, 1], ":", c="r", lw=0.8, alpha=0.4)

        # dynamic artists
        (self.tar_trail,) = ax.plot([], [], "-", c="r", lw=1.2, alpha=0.6)
        (self.tar_pt,) = ax.plot([], [], "*", ms=16, mfc="#e11", mec="#7a0000", zorder=12)
        self.u = []
        for j in range(n):
            c = COLORS[j % len(COLORS)]
            a = {
                "fov": ax.add_patch(Rectangle((0, 0), 2 * self.VR, 2 * self.VR, fc=c, ec=c,
                                              alpha=0.12, lw=0.8, zorder=2)),
                "corr": ax.add_patch(Polygon(np.zeros((3, 2)), closed=True, fc=c, ec=c,
                                             alpha=0.15, lw=1.0, ls="--", zorder=3)),
                "trail": ax.plot([], [], "-", c=c, lw=1.2, alpha=0.8, zorder=4)[0],
                "ref": ax.plot([], [], "-", c=c, lw=0.8, alpha=0.5, zorder=4)[0],
                "pred": ax.plot([], [], ".-", c=c, lw=1.4, ms=4, zorder=6)[0],
                # markers keep their screen size at any zoom; "rad" is the true radius
                "body": ax.plot([], [], "o", ms=8, mfc=c, mec="k", mew=0.8, zorder=8)[0],
                "rad": ax.add_patch(Circle((0, 0), self.R, fc="none", ec=c, lw=0.8, zorder=8)),
                "lead": ax.plot([], [], "o", ms=18, mfc="none", mec=LEADER_COLOR, mew=2.5,
                                zorder=7)[0],
                "fail": ax.plot([], [], "x", c="red", ms=14, mew=3, zorder=9)[0],
                "label": ax.text(0, 0, f"U{self.ids[j]}", fontsize=9, color=c, zorder=10,
                                 fontweight="bold"),
            }
            self.u.append(a)
        self.txt = self.info.text(0, 1, "", va="top", family="monospace", fontsize=9)

        sax = self.fig.add_axes([0.08, 0.025, 0.55, 0.03])
        self.slider = Slider(sax, "step", 0, self.T - 1, valinit=self.k, valstep=1)
        self._from_code = False
        self.slider.on_changed(self.on_slider)

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.manager.set_window_title(f"debug_viewer — {run_dir}")
        print(__doc__)
        print(f"[VIEW] {run_dir}: {self.T} steps, {n} UAVs, "
              f"{len(self.fails)} steps with an MPC failure")
        if self.pred[0] is None:
            print("[VIEW] This run has no MPC predictions / modes (recorded by newer runs only).")
        self.draw()

    # ---------- per-step drawing ----------
    def draw(self):
        k, ax = self.k, self.ax
        lo = max(0, k - TRAIL) if self.show["trail"] else k
        tx, ty = self.tar[k]
        self.tar_pt.set_data([tx], [ty])
        self.tar_trail.set_data(self.tar[lo:k + 1, 0], self.tar[lo:k + 1, 1])
        if self.follow:
            box = (tx - self.W, tx + self.W, ty - self.W, ty + self.W)
        else:
            box = (*self.xlim, *self.ylim)
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])

        lines = [f"step {k} / {self.T - 1}   t = {k * self.dt:.1f} s",
                 f"play x{self.speed:g}{'  >' if self.playing else '  (paused)'}"]
        if not self.playing:
            lines += [f"target  ({tx:7.1f}, {ty:7.1f})",
                      f"view    {'follow ±%.0f m' % self.W if self.follow else 'whole map'}", "",
                      " UAV  mode    d_tgt   |v|   solve  MPC",
                      " ---  ------  -----  -----  -----  ----"]
        for j, a in enumerate(self.u):
            P = self.paths[j]
            x, y = P[k, 1], P[k, 2]
            a["body"].set_data([x], [y])
            a["rad"].center = (x, y)
            a["label"].set_position((x + 1.5 * self.R, y + 1.5 * self.R))
            a["fov"].set_xy((x - self.VR, y - self.VR))
            a["trail"].set_data(P[lo:k + 1, 1], P[lo:k + 1, 2])

            st = self.status[j][k] if self.status[j] is not None and k < len(self.status[j]) else None
            leader = st is not None and bool(st[1])
            failed = st is not None and st[2] > 0
            a["lead"].set_data([x] if leader else [], [y] if leader else [])
            a["fail"].set_data([x] if failed else [], [y] if failed else [])

            pr = self.pred[j][k] if self.pred[j] is not None and k < len(self.pred[j]) else None
            if self.show["pred"] and pr is not None:
                a["pred"].set_data(pr[:, 0], pr[:, 1])
            else:
                a["pred"].set_data([], [])

            ref = self.refs[j][k] if self.refs[j] is not None and k < len(self.refs[j]) else None
            if self.show["ref"] and ref is not None and np.ndim(ref) == 2:
                a["ref"].set_data(ref[:, 0], ref[:, 1])
            else:
                a["ref"].set_data([], [])

            poly = None
            c = self.corr[j][k] if self.corr[j] is not None and k < len(self.corr[j]) else None
            if self.show["corr"] and isinstance(c, dict) and c.get("A") is not None and len(c["A"]):
                pad = 10 * self.W
                poly = halfspace_polygon(c["A"][0], c["b"][0],
                                         (x - pad, x + pad, y - pad, y + pad))
            a["corr"].set_visible(poly is not None)
            if poly is not None:
                a["corr"].set_xy(poly)

            d = np.hypot(x - tx, y - ty)
            v = np.hypot(P[k, 4], P[k, 5])
            mode = "-" if st is None else ("TRACK" if st[0] else "SEARCH")
            role = "L" if leader else " "
            ms = f"{1000 * self.ct[k, j]:5.0f}" if self.ct is not None and k < len(self.ct) else "    -"
            mpc = "-" if st is None else ("FAIL" if failed else "ok")
            if not self.playing:
                lines.append(f" U{self.ids[j]:<2} {mode:<6}{role} {d:6.1f} {v:6.2f}  {ms}  {mpc}")

        # the full table costs ~half of each frame to draw: shown only when paused
        if not self.playing:
            nxt = self.fails[self.fails > k]
            lines += ["", f"MPC failures: {len(self.fails)} steps"
                          + (f", next at {nxt[0]}" if len(nxt) else "")]
            lines += ["", "space play  ←/→ ±1  ↑/↓ ±10  PgUp/PgDn ±100",
                      "n/b next/prev failure   +/- zoom   f follow",
                      "c corridor  m MPC pred  r ref  t trails  [ ] speed"]
        self.txt.set_text("\n".join(lines))
        ax.set_title("solid dots = MPC plan, thin = planner ref, dashed = corridor, "
                     "yellow ring = leader, red x = MPC failed", fontsize=8)

        self._from_code = True
        self.slider.set_val(k)
        self._from_code = False
        self.fig.canvas.draw_idle()

    # ---------- interaction ----------
    def goto(self, k):
        self.k = int(np.clip(k, 0, self.T - 1))
        self.draw()

    def on_slider(self, val):
        if not self._from_code:
            self.goto(val)

    def tick(self):
        if self.k >= self.T - 1:
            self.toggle_play()
            return
        now = time.perf_counter()
        self._carry += (now - self._t_last) * self.speed / self.dt
        self._t_last = now
        n = int(self._carry)
        if n:
            self._carry -= n
            self.goto(self.k + n)

    def toggle_play(self):
        self.playing = not self.playing
        self._t_last, self._carry = time.perf_counter(), 0.0
        (self.timer.start if self.playing else self.timer.stop)()
        self.draw()

    def on_key(self, e):
        steps = {"right": 1, "left": -1, "up": 10, "down": -10, "pageup": 100, "pagedown": -100}
        key = e.key
        if key == " ":
            self.toggle_play()
        elif key in steps:
            self.goto(self.k + steps[key])
        elif key == "home":
            self.goto(0)
        elif key == "end":
            self.goto(self.T - 1)
        elif key == "n":
            nxt = self.fails[self.fails > self.k]
            if len(nxt):
                self.goto(nxt[0])
        elif key == "b":
            prv = self.fails[self.fails < self.k]
            if len(prv):
                self.goto(prv[-1])
        elif key in ("+", "="):
            self.W = max(2.0, self.W / 1.5); self.draw()
        elif key in ("-", "_"):
            self.W *= 1.5; self.draw()
        elif key == "f":
            self.follow = not self.follow; self.draw()
        elif key == "]":
            self.speed = min(self.speed * 2, 256); self.draw()
        elif key == "[":
            self.speed = max(self.speed / 2, 0.125); self.draw()
        elif key in ("c", "m", "r", "t"):
            name = {"c": "corr", "m": "pred", "r": "ref", "t": "trail"}[key]
            self.show[name] = not self.show[name]; self.draw()
        elif key in ("q", "escape"):
            plt.close(self.fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run directory or its data.pkl (default: latest in runs/)")
    p.add_argument("--start", type=int, default=0, help="first step shown")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed, x real time (default 1; [ / ] change it while running)")
    p.add_argument("--window", type=float, default=None,
                   help="half size of the view around the target in m (default 4 * VIEWING_RADIUS)")
    a = p.parse_args()
    viewer = Viewer(find_run(a.run), a.start, a.window, a.speed)   # keep a reference (weak mpl callbacks)
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
