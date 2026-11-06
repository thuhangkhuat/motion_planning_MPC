"""
plot_paper.py — Figures for the paper, from one run directory (data.pkl +
config.json only, so a run made on another machine or branch can be drawn).

    python plot_paper.py scenario runs/<dir>                  # map, starts, target path
    python plot_paper.py overview runs/<dir> --t 120 300 450  # whole run + the snapshot windows
    python plot_paper.py snap runs/<dir> --t 120 300 450      # snapshots at those times

Common options: --out <path without extension> (default results/paper/<paper scen>_<kind>, see PAPER_NAME),
--width <inches> (3.5 = one column, 7.16 = two columns), --labels (U0.., W0..).
Each figure is written as .pdf and .png.

  scenario  obstacles (black), map border, UAV starts, target path (its
            waypoints only with --waypoints); legs faster than 1.2 x the
            nominal target speed are drawn as sprint legs
  overview  every UAV path (thin), target path (time ticks with --tick-every), leader
            hand-offs, final positions; --t t1 t2 .. (the same times as for
            snap) outlines the snapshot windows (a), (b), .. so that the
            labels of the two figures match
  snap      one panel per time: trails of the last --trail seconds (fading),
            UAVs and their FOV squares at that time, leader ring, target;
            window of half side --half metres around the target
"""

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
})

OBS_COLOR = "black"
TARGET_COLOR = "#d62728"
SPRINT_COLOR = "#ff7f0e"
FOV_COLOR = "#2ca02c"
LEADER_COLOR = "#f1c40f"
UAV_COLORS = [c for k, c in enumerate(plt.get_cmap("tab10").colors) if k != 3]   # red = target
SPRINT_RATIO = 1.2          # as evaluate.py: a leg faster than 1.2 x the nominal speed


# ============================================================
# Loading
# ============================================================
def load(run_dir):
    with open(os.path.join(run_dir, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    sd, prm = cfg.get("scenario_def", {}), cfg.get("params", {})
    ids = sorted(k for k in data if isinstance(k, int))
    T = min(len(data[0]["tar_traj"]), *[len(data[i]["path"]) for i in ids])
    run = {
        "name": sd.get("name", os.path.basename(os.path.normpath(run_dir)).split("_")[0]),
        "sd": sd, "cfg": cfg,
        "dt": float(prm.get("TIMESTEP", 0.1)),
        "VR": float(data.get("meta", {}).get("fov_side") or 2 * sd.get("viewing_radius", 25.0)) / 2,
        "P": np.stack([np.asarray(data[i]["path"])[:T, 1:3] for i in ids], 1),   # (T, n, 2)
        "t": np.asarray(data[ids[0]]["path"])[:T, 0],
        "tar": np.asarray(data[0]["tar_traj"])[:T, :2],
        "n": len(ids),
    }
    st = [data[i].get("status") for i in ids]
    run["leader"] = (np.stack([np.asarray(s)[:T, 1] > 0 for s in st], 1)
                     if all(s is not None and len(s) >= T for s in st) else None)
    return run


def step_at(run, t):
    return int(np.clip(np.searchsorted(run["t"], t), 0, len(run["t"]) - 1))


def leader_seq(run):
    """Leader index per step (status flag; nearest UAV to the target if not recorded)."""
    if run["leader"] is not None:
        L = run["leader"]
        idx = np.where(L.any(1), L.argmax(1), -1)
        for k in range(1, len(idx)):                     # no leader flagged: keep the last one
            if idx[k] < 0:
                idx[k] = idx[k - 1]
        return idx
    return np.linalg.norm(run["P"] - run["tar"][:, None], axis=2).argmin(1)


# ============================================================
# Drawing helpers
# ============================================================
def draw_map(ax, sd, border=True):
    for x, y, w, h in sd.get("rects") or []:
        ax.add_patch(patches.Rectangle((x, y), w, h, fc=OBS_COLOR, ec=OBS_COLOR, lw=0.3, zorder=3))
    for cx, cy, r in sd.get("circles") or []:
        ax.add_patch(patches.Circle((cx, cy), r, fc=OBS_COLOR, ec=OBS_COLOR, lw=0.3, zorder=3))
    if border:
        (x0, x1), (y0, y1) = sd["xlim"], sd["ylim"]
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="black",
                                       lw=0.8, zorder=4))


def setup_axes(ax, xlim, ylim, grid=True):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if grid:
        ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(width=0.5, length=2)


def fig_for(sd, width, extra_h=0.0):
    (x0, x1), (y0, y1) = sd["xlim"], sd["ylim"]
    return plt.subplots(figsize=(width, width * (y1 - y0) / (x1 - x0) + extra_h))


def fading_line(ax, xy, color, lw, zorder, a0=0.05, a1=1.0):
    """Polyline whose alpha grows from a0 (oldest) to a1 (newest)."""
    if len(xy) < 2:
        return
    seg = np.stack([xy[:-1], xy[1:]], 1)
    rgba = np.tile(to_rgba(color), (len(seg), 1))
    rgba[:, 3] = np.linspace(a0, a1, len(seg))
    ax.add_collection(LineCollection(seg, colors=rgba, linewidths=lw, zorder=zorder,
                                     capstyle="round"))


def draw_target(ax, xy, size=9, zorder=12):
    ax.plot(*xy, marker="*", ms=size, mfc=TARGET_COLOR, mec="black", mew=0.5, zorder=zorder)


def draw_uav(ax, xy, color, leader=False, size=4.5, zorder=11):
    ax.plot(*xy, "o", ms=size, mfc=color, mec="black", mew=0.5, zorder=zorder)
    if leader:
        ax.plot(*xy, "o", ms=size * 2.1, mfc="none", mec=LEADER_COLOR, mew=1.4, zorder=zorder - 1)


def draw_scalebar(ax, length):
    """Horizontal bar of `length` m in the lower-left corner, labelled."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x, y = x0 + 0.05 * (x1 - x0), y0 + 0.05 * (y1 - y0)
    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    ax.plot([x, x + length], [y, y], color="black", lw=1.5, solid_capstyle="butt", zorder=13,
            path_effects=halo)
    ax.text(x + length / 2, y + 0.015 * (y1 - y0), f"{length:g} m", ha="center", va="bottom",
            fontsize=7, zorder=13, path_effects=halo)


def draw_fov_swept(ax, run, res=1.0, every=5):
    """Every ground point that was inside some UAV's FOV at some time, as one
    flat light area (a raster mask, so overlaps do not darken)."""
    (x0, x1), (y0, y1) = run["sd"]["xlim"], run["sd"]["ylim"]
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    seen = np.zeros((ny, nx), bool)
    VR = run["VR"]
    for x, y in run["P"][::every].reshape(-1, 2):
        i0, i1 = int((x - VR - x0) / res), int(np.ceil((x + VR - x0) / res))
        j0, j1 = int((y - VR - y0) / res), int(np.ceil((y + VR - y0) / res))
        seen[max(j0, 0):max(j1, 0), max(i0, 0):max(i1, 0)] = True
    rgba = np.zeros((ny, nx, 4))
    rgba[seen] = to_rgba(FOV_COLOR, 0.22)
    ax.imshow(rgba, origin="lower", extent=(x0, x0 + nx * res, y0, y0 + ny * res),
              interpolation="nearest", zorder=1)


def sprint_mask(run):
    """Per step: the target is on a leg faster than SPRINT_RATIO x the nominal speed."""
    v = np.linalg.norm(np.diff(run["tar"], axis=0), axis=1) / run["dt"]
    v = np.r_[v, v[-1:]]
    nominal = np.median(v[v > 0.1]) if (v > 0.1).any() else 0.0
    return v > SPRINT_RATIO * nominal


# paper numbering of the scenarios (scen5 is not in the paper)
PAPER_NAME = {"scen1": "scen1", "scen2": "scen3", "scen3": "scen6", "scen4": "scen2",
              "scen6": "scen4", "scen7": "scen5", "scen5": "unused_scen5"}


def save(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"[PLOT] wrote {out}.pdf / .png")


def split_runs(mask):
    """(start, end) index pairs of the True runs of a boolean array."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# ============================================================
# Figures
# ============================================================
def plot_scenario(run, a):
    sd = run["sd"]
    fig, ax = fig_for(sd, a.width)
    draw_map(ax, sd)
    tar = run["tar"]
    ax.plot(tar[:, 0], tar[:, 1], color=TARGET_COLOR, lw=1.0, zorder=5)
    sprint = sprint_mask(run)
    for s, e in split_runs(sprint):
        ax.plot(tar[s:e + 1, 0], tar[s:e + 1, 1], color=SPRINT_COLOR, lw=1.0, zorder=6,
                solid_capstyle="round")
    wp = np.array([w[:2] for w in sd.get("waypoints") or []]).reshape(-1, 2)
    if a.waypoints and len(wp):
        ax.plot(wp[:, 0], wp[:, 1], "o", ms=2.5, mfc="white", mec=TARGET_COLOR, mew=0.7, zorder=7)
        if a.labels:
            for i, (x, y) in enumerate(wp):
                ax.annotate(f"W{i}", (x, y), xytext=(3, 3), textcoords="offset points",
                            fontsize=6, color=TARGET_COLOR)
    draw_target(ax, tar[0])
    # ax.plot(*tar[-1], marker="X", ms=7, mfc=TARGET_COLOR, mec="black", mew=0.5, zorder=12)
    st = np.asarray(sd.get("starts") or run["P"][0], float)[:, :2]
    ax.plot(st[:, 0], st[:, 1], "^", ms=4, mfc="blue", mec="black", mew=0.5, zorder=8)
    if a.labels:
        for i, (x, y) in enumerate(st):
            ax.annotate(f"U{i}", (x, y), xytext=(3, -7), textcoords="offset points", fontsize=6)
    setup_axes(ax, sd["xlim"], sd["ylim"], grid=False)
    h = [ #Line2D([], [], color=OBS_COLOR, marker="s", ls="", ms=5, label="Obstacle"),
        #  Line2D([], [], color=TARGET_COLOR, lw=1.0, label="Target path"),
        #  Line2D([], [], marker="*", ls="", ms=7, mfc=TARGET_COLOR, mec="black", mew=0.5,
                # label="Target start"),
        #  Line2D([], [], marker="X", ls="", ms=6, mfc=TARGET_COLOR, mec="black", mew=0.5,
        #         label="Target goal"),
        #  Line2D([], [], marker="^", ls="", ms=4, mfc="blue", mec="black", mew=0.5,
        #         label="UAV start")
                ]
    # if sprint.any():
    #     h.insert(2, Line2D([], [], color=SPRINT_COLOR, lw=2.2, label="Sprint leg"))
    if h:
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    return fig


def plot_overview(run, a):
    sd, P, tar, dt = run["sd"], run["P"], run["tar"], run["dt"]
    fig, ax = fig_for(sd, a.width)
    draw_map(ax, sd)
    if a.fov == "swept":
        draw_fov_swept(ax, run)
    if a.paths:
        for i in range(run["n"]):
            ax.plot(P[:, i, 0], P[:, i, 1], color=UAV_COLORS[i % len(UAV_COLORS)], lw=0.5, alpha=0.7, zorder=5)
    ax.plot(tar[:, 0], tar[:, 1], color=TARGET_COLOR, lw=0.8, zorder=6)
    # time ticks on the target path
    every = a.tick_every
    if every > 0:
        for k in range(int(round(every / dt)), len(tar), int(round(every / dt))):
            ax.plot(*tar[k], "o", ms=2.2, mfc="white", mec=TARGET_COLOR, mew=0.6, zorder=7)
            ax.annotate(f"{run['t'][k]:.0f} s", tar[k], xytext=(3, 2), textcoords="offset points",
                        fontsize=5.5, color=TARGET_COLOR, zorder=7)
    # leader hand-offs
    ld = leader_seq(run)
    hand = np.flatnonzero(np.diff(ld) != 0) + 1
    ax.plot(tar[hand, 0], tar[hand, 1], "D", ms=3, mfc=LEADER_COLOR, mec="black", mew=0.4, zorder=8)
    st = P[0]
    ax.plot(st[:, 0], st[:, 1], "^", ms=4, mfc="blue", mec="black", mew=0.5, zorder=8)
    for i in range(run["n"]):
        if a.fov == "final":
            VR = run["VR"]
            ax.add_patch(patches.Rectangle(P[-1, i] - VR, 2 * VR, 2 * VR, fc=FOV_COLOR, ec=FOV_COLOR,
                                           alpha=0.25, lw=0.5, zorder=2))
        draw_uav(ax, P[-1, i], UAV_COLORS[i % len(UAV_COLORS)], leader=False, size=3.2)
    draw_target(ax, tar[-1], size=7)
    for j, t in enumerate(a.t or []):
        c = tar[step_at(run, t)]
        ax.add_patch(patches.Rectangle(c - a.half, 2 * a.half, 2 * a.half, fc="none", ec="#1f4e79",
                                       lw=0.9, ls="-", zorder=9))
        # small label just above the top-left corner, outside the window; a thin
        # white outline keeps it readable where it crosses a (black) obstacle
        if len(a.t) > 1:                                 # one window: no (a) label
            ax.annotate(f"({chr(97 + j)})", c + [-a.half, a.half], xytext=(0, 0.5),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=a.box_label_size, color="#1f4e79", zorder=10,
                        path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])
    setup_axes(ax, sd["xlim"], sd["ylim"], grid=False)
    h = [#Line2D([], [], color=OBS_COLOR, marker="s", ls="", ms=5, label="Obstacle"),
        #  Line2D([], [], color="gray", lw=0.8, label="UAV paths"),
        #  Line2D([], [], color=TARGET_COLOR, lw=1.1, ls="--", label="Target path"),
        #  Line2D([], [], marker="D", ls="", ms=3, mfc=LEADER_COLOR, mec="black", mew=0.4,
                # label="Leader hand-off"),
         #Line2D([], [], marker="^", ls="", ms=4, mfc="white", mec="black", mew=0.5, label="UAV start"),
        #  Line2D([], [], marker="o", ls="", ms=6.5, mfc="none", mec=LEADER_COLOR, mew=1.4,
        #         label="Leader")
                ]
    if not a.paths:
        h = h[1:]                                        # no "UAV paths" entry
    # if a.fov != "none":
    #     h.append(Line2D([], [], marker="s", ls="", ms=6, mfc=to_rgba(FOV_COLOR, 0.3), mec="none",
    #                     label="FOV" if a.fov == "swept" else "FOV (final)"))
    if h:
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    return fig


def plot_snap(run, a):
    times = a.t
    if not times:
        raise SystemExit("[PLOT] snap needs --t t1 t2 ...")
    sd, P, tar, dt, VR = run["sd"], run["P"], run["tar"], run["dt"], run["VR"]
    cols = a.cols or len(times)
    rows = int(np.ceil(len(times) / cols))
    w = a.width / cols
    fig, axes = plt.subplots(rows, cols, figsize=(a.width, rows * w + 0.35), squeeze=False)
    ld = leader_seq(run)
    nt = int(round(a.trail / dt))
    for j, t in enumerate(times):
        ax = axes.flat[j]
        k = step_at(run, t)
        c = tar[k] if a.center == "target" else P[k].mean(0)
        draw_map(ax, sd)
        k0 = max(0, k - nt)
        fading_line(ax, tar[k0:k + 1], TARGET_COLOR, 1.0, 6)
        for i in range(run["n"]):
            col = UAV_COLORS[i % len(UAV_COLORS)]
            fading_line(ax, P[k0:k + 1, i], col, 0.9, 7)
            x, y = P[k, i]
            ax.add_patch(patches.Rectangle((x - VR, y - VR), 2 * VR, 2 * VR, fc=FOV_COLOR, ec=FOV_COLOR,
                                           alpha=0.10, lw=0, zorder=2))
            ax.add_patch(patches.Rectangle((x - VR, y - VR), 2 * VR, 2 * VR, fc="none", ec=FOV_COLOR,
                                           alpha=0.8, lw=0.5, zorder=2))
            draw_uav(ax, (x, y), col, leader=(i == ld[k]))
        draw_target(ax, tar[k])
        ax.set_xlim(c[0] - a.half, c[0] + a.half)
        ax.set_ylim(c[1] - a.half, c[1] + a.half)
        ax.set_aspect("equal")
        ax.set_xticks([])                                # no axis labels / numbers: a scale bar instead
        ax.set_yticks([])
        if a.scalebar > 0:
            draw_scalebar(ax, a.scalebar)
        ax.set_title((f"({chr(97 + j)}) " if len(times) > 1 else "") + f"$t$ = {run['t'][k]:.0f} s", pad=3, fontsize=a.title_size)
        # if j % cols:
        #     ax.set_ylabel("")
        # if j < len(times) - cols:
        #     ax.set_xlabel("")
    for ax in axes.flat[len(times):]:
        ax.axis("off")
    h = [#Line2D([], [], color=OBS_COLOR, marker="s", ls="", ms=5, label="Obstacle"),
        #  Line2D([], [], color="gray", lw=0.9, label=f"UAV trail (last {a.trail:.0f} s)"),
        #  Line2D([], [], color=TARGET_COLOR, lw=1.0, label="Target trail"),
        #  Line2D([], [], marker="*", ls="", ms=7, mfc=TARGET_COLOR, mec="black", mew=0.5, label="Target"),
        #  Line2D([], [], marker="s", ls="", ms=6, mfc=to_rgba(FOV_COLOR, 0.2), mec=FOV_COLOR, mew=0.5,
        #         label="FOV"),
        #  Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec=LEADER_COLOR, mew=1.4, label="Leader")
        ]
    if h:
        fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05 if h else 0, 1, 1), w_pad=0.6)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["scenario", "overview", "snap"])
    ap.add_argument("run", help="run directory (runs/<dir>)")
    ap.add_argument("--out", default=None, help="output path without extension")
    ap.add_argument("--width", type=float, default=None,
                    help="figure width in inches (default 3.5; snap 7.16)")
    ap.add_argument("--labels", action="store_true", help="label UAV starts / waypoints")
    ap.add_argument("--waypoints", action="store_true", help="scenario: also mark the target waypoints")
    ap.add_argument("--t", "--boxes", dest="t", type=float, nargs="*",
                    help="snapshot times [s]: the panels of snap, the windows outlined in overview")
    ap.add_argument("--half", type=float, default=100.0, help="snapshot window half side [m]")
    ap.add_argument("--title-size", type=float, default=8.0,
                    help="snap: font size of the (a) t = .. s panel titles [pt]")
    ap.add_argument("--box-label-size", type=float, default=6.0,
                    help="overview: font size of the (a), (b), .. window labels [pt]")
    ap.add_argument("--trail", type=float, default=25.0, help="snap: trail length [s]")
    ap.add_argument("--scalebar", type=float, default=0.0, help="snap: scale bar length [m] (0 = off)")
    ap.add_argument("--cols", type=int, default=None, help="snap: panels per row (default all in one row)")
    ap.add_argument("--center", choices=["target", "team"], default="target", help="snap: window centre")
    ap.add_argument("--no-paths", dest="paths", action="store_false",
                    help="overview: do not draw the UAV paths (e.g. with --fov swept)")
    ap.add_argument("--fov", choices=["none", "final", "swept"], default="none",
                    help="overview: no FOV, the FOVs at the final time, or the whole area the FOVs swept")
    ap.add_argument("--tick-every", type=float, default=0.0,
                    help="overview: time tick on the target path every N s (0 = off)")
    ap.add_argument("--show", action="store_true", help="also open the figure")
    a = ap.parse_args()
    if a.width is None:
        a.width = 7.16 if a.kind == "snap" else 3.5
    run = load(a.run)
    fig = {"scenario": plot_scenario, "overview": plot_overview, "snap": plot_snap}[a.kind](run, a)
    save(fig, a.out or os.path.join("results", "paper", f"{PAPER_NAME.get(run['name'], run['name'])}_{a.kind}"))
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
