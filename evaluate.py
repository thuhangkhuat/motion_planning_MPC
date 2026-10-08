"""
evaluate.py — Score finished runs on the benchmark scenarios.

    python evaluate.py                      # latest run of each benchmark (scen1..scen4)
    python evaluate.py runs/<dir> ...       # given runs (any scenario)
    python evaluate.py --out results/bench  # where the tables go (default results/benchmark)

Everything is computed from runs/<dir>/data.pkl, config.json and log.txt, so a
run can be scored on another machine than the one that made it. Writes
<out>.md (one table per metric group) and <out>.csv (one row per run), and
prints the Markdown table.

Metrics (t = all steps; "after formation" = from the first step where every
UAV is in TRACK, so the gathering at the start is not counted):
  Tracking   target seen (true target inside some UAV's square FOV), lost
             episodes and the longest one, target offset in the leader's FOV
  Leader     hand-offs, how many fall in a sprint window (target faster than
             VMAX, + SPRINT_TAIL s), steps with no leader / several leaders
  Formation  FOV union area / n FOVs (1 = no overlap: FOVs side by side),
             steps with a UAV pair closer than CLOSE_PAIR m
  Safety     min UAV-UAV distance, min clearance to obstacles (negative =
             inside an obstacle), steps outside the map
  Solver     MPC failures, IPOPT fallbacks / emergency stops / planner
             failures from the log, compute time per UAV per step
  Effort     mean |u| and path length per UAV
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re

import numpy as np

BENCHMARKS = ["scen1", "scen2", "scen3", "scen4"]
CLOSE_PAIR = 10.0      # m: "two UAVs on top of each other" (FOV half side is 25 m)
SPRINT_TAIL = 3.0      # s after a sprint in which a hand-off still counts as caused by it
FORMATION_EVERY = 5    # FOV union area is computed every N steps (exact, but per step)


# ============================================================
# Loading
# ============================================================
def latest_run(scenario):
    runs = sorted(glob.glob(os.path.join("runs", f"{scenario}_*")), key=os.path.getmtime)
    runs = [r for r in runs if os.path.isfile(os.path.join(r, "data.pkl"))]
    return runs[-1] if runs else None


def load(run_dir):
    with open(os.path.join(run_dir, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    log = ""
    if os.path.isfile(os.path.join(run_dir, "log.txt")):
        with open(os.path.join(run_dir, "log.txt"), encoding="utf-8", errors="replace") as f:
            log = f.read()
    return data, cfg, log


# ============================================================
# Geometry helpers
# ============================================================
def union_area(centres, half):
    """Exact area of the union of axis-aligned squares (coordinate compression)."""
    xs = np.unique(np.r_[centres[:, 0] - half, centres[:, 0] + half])
    ys = np.unique(np.r_[centres[:, 1] - half, centres[:, 1] + half])
    cx, cy = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    inside = np.zeros((len(cy), len(cx)), bool)
    for x, y in centres:
        inside |= (np.abs(cx[None] - x) <= half) & (np.abs(cy[:, None] - y) <= half)
    return float((np.diff(ys)[:, None] * np.diff(xs)[None] * inside).sum())


def obstacle_clearance(P, rects, circles):
    """Distance from each point (N, 2) to the nearest obstacle (negative inside)."""
    d = np.full(len(P), np.inf)
    for x, y, w, h in rects:
        qx = np.abs(P[:, 0] - (x + w / 2)) - w / 2
        qy = np.abs(P[:, 1] - (y + h / 2)) - h / 2
        out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        d = np.minimum(d, out + np.minimum(np.maximum(qx, qy), 0))
    for cx, cy, r in circles:
        d = np.minimum(d, np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)
    return d


def runs_of(mask):
    """(start, length) of each run of True in a boolean array."""
    m = np.r_[False, mask, False].astype(int)
    starts, ends = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    return list(zip(starts, ends - starts))


# ============================================================
# Evaluation of one run
# ============================================================
def evaluate(run_dir):
    data, cfg, log = load(run_dir)
    sd, prm = cfg.get("scenario_def", {}), cfg.get("params", {})
    ids = sorted(k for k in data if isinstance(k, int))
    n = len(ids)
    T = min(len(data[0]["tar_traj"]), *[len(data[i]["path"]) for i in ids])
    dt = float(prm.get("TIMESTEP", 0.1))
    VR = float(sd.get("viewing_radius", prm.get("VIEWING_RADIUS", 1.0)))
    R = float(prm.get("ROBOT_RADIUS", 0.5))
    vmax = float(prm.get("VMAX", 10.0))
    cbf = float(prm.get("CBF_BOX_RATIO", float("nan")))
    xlim, ylim = sd.get("xlim", [-np.inf, np.inf]), sd.get("ylim", [-np.inf, np.inf])

    path = np.stack([np.asarray(data[i]["path"])[:T] for i in ids], 1)    # (T, n, 13)
    P, V, U = path[:, :, 1:3], path[:, :, 4:6], path[:, :, 7:9]
    tar = np.asarray(data[0]["tar_traj"])[:T, :2]
    has_status = all(data[i].get("status") is not None and len(data[i]["status"]) >= T for i in ids)
    st = (np.stack([np.asarray(data[i]["status"])[:T] for i in ids], 1)
          if has_status else np.zeros((T, n, 3), int))
    track, lead, fail = st[:, :, 0] > 0, st[:, :, 1] > 0, st[:, :, 2] > 0

    ready = int(np.argmax(track.all(1))) if track.all(1).any() else 0
    after = np.arange(T) >= ready
    r = {"run": os.path.basename(os.path.normpath(run_dir)),
         "scenario": sd.get("name", data.get("meta", {}).get("scenario", "?")),
         "uavs": n, "steps": T, "duration_s": round(T * dt, 1),
         "finished": bool(data.get("meta", {}).get("finished", False)),
         "CBF_BOX_RATIO": cbf, "formation_ready_s": round(ready * dt, 1)}

    # ── tracking ──
    linf = np.abs(P - tar[:, None]).max(-1)                 # (T, n) target offset per UAV
    seen = (linf <= VR).any(1)
    lost = [L for s, L in runs_of(~seen) if s >= ready]
    r["seen_%"] = round(100 * seen[after].mean(), 2)
    r["lost_episodes"] = len(lost)
    r["longest_loss_s"] = round(max(lost, default=0) * dt, 1)
    has_lead = lead.any(1)
    off = np.where(has_lead, linf[np.arange(T), np.argmax(lead, 1)], np.nan)
    sel = after & has_lead
    r["leader_offset_p50_m"] = round(float(np.median(off[sel])), 1) if sel.any() else np.nan
    r["leader_offset_p95_m"] = round(float(np.percentile(off[sel], 95)), 1) if sel.any() else np.nan

    # ── leader ──
    who = np.where(has_lead, np.argmax(lead, 1), -1)
    seq = []                                                # (uav, first step) per leader term
    for t in np.flatnonzero(who >= 0):
        if not seq or seq[-1][0] != who[t]:
            seq.append((int(who[t]), int(t)))
    changes = [t for (_, t) in seq[1:]]
    tspeed = np.r_[0, np.linalg.norm(np.diff(tar, axis=0), axis=1) / dt]
    sprint = tspeed > vmax
    tail = int(round(SPRINT_TAIL / dt))
    win = np.zeros(T, bool)
    for s, L in runs_of(sprint):
        win[s:min(T, s + L + tail)] = True
    r["sprints"] = len(runs_of(sprint))
    r["handoffs"] = len(changes)
    r["handoffs_in_sprint"] = int(sum(win[t] for t in changes))
    r["handoffs_outside_sprint"] = r["handoffs"] - r["handoffs_in_sprint"]
    r["no_leader_steps"] = int((~has_lead & seen)[after].sum())
    r["multi_leader_steps"] = int((lead.sum(1) > 1)[after].sum())
    r["leader_sequence"] = " ".join(f"U{u}@{t * dt:.0f}s" for u, t in seq)

    # ── formation ──
    D = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1) + np.eye(n) * 1e9
    dmin = D.min(axis=(1, 2))
    r["close_pair_%"] = round(100 * (dmin < CLOSE_PAIR)[after].mean(), 1)
    ks = np.arange(ready, T, FORMATION_EVERY)
    if n > 1 and len(ks):
        ratio = np.array([union_area(P[k], VR) for k in ks]) / (n * (2 * VR) ** 2)
        r["fov_union_ratio_mean"] = round(float(ratio.mean()), 3)
    else:
        r["fov_union_ratio_mean"] = np.nan

    # ── safety ──
    r["min_uav_dist_m"] = round(float(dmin.min()), 2) if n > 1 else np.nan
    r["uav_collision_steps"] = int((dmin < 2 * R).sum()) if n > 1 else 0
    clr = obstacle_clearance(P.reshape(-1, 2), sd.get("rects") or [],
                             sd.get("circles") or []).reshape(T, n) - R
    r["min_obstacle_clearance_m"] = round(float(clr.min()), 2) if np.isfinite(clr).any() else np.nan
    r["obstacle_collision_steps"] = int((clr < 0).any(1).sum())
    out = ((P[:, :, 0] < xlim[0]) | (P[:, :, 0] > xlim[1])
           | (P[:, :, 1] < ylim[0]) | (P[:, :, 1] > ylim[1])).any(1)
    r["outside_map_steps"] = int(out.sum())

    # ── solver ──
    r["mpc_fail_steps"] = int(fail.sum()) if has_status else np.nan
    r["mpc_fail_%"] = round(100 * fail.mean(), 2) if has_status else np.nan
    r["emergency_stops"] = len(re.findall(r"Emergency stop", log))
    r["planner_failures"] = len(re.findall(r"JPS failed|plan(?:ner)? failed", log, re.I))
    ct = np.asarray(data.get("meta", {}).get("compute_times", []), float)
    r["compute_ms_mean"] = round(1e3 * ct.mean(), 1) if ct.size else np.nan
    r["compute_ms_p95"] = round(1e3 * np.percentile(ct, 95), 1) if ct.size else np.nan

    # ── effort ──
    r["mean_accel_cmd"] = round(float(np.linalg.norm(U, axis=-1).mean()), 3)
    r["path_length_per_uav_m"] = round(float(np.linalg.norm(np.diff(P, axis=0), axis=-1).sum(0).mean()), 1)
    r["mean_speed"] = round(float(np.linalg.norm(V, axis=-1).mean()), 2)
    return r


# ============================================================
# Output
# ============================================================
GROUPS = [
    ("Run", ["scenario", "run", "uavs", "duration_s", "finished", "CBF_BOX_RATIO", "formation_ready_s"]),
    ("Tracking", ["seen_%", "lost_episodes", "longest_loss_s", "leader_offset_p50_m", "leader_offset_p95_m"]),
    ("Leader", ["sprints", "handoffs", "handoffs_in_sprint", "handoffs_outside_sprint",
                "no_leader_steps", "multi_leader_steps"]),
    ("Formation", ["fov_union_ratio_mean", "close_pair_%"]),
    ("Safety", ["min_uav_dist_m", "uav_collision_steps", "min_obstacle_clearance_m",
                "obstacle_collision_steps", "outside_map_steps"]),
    ("Solver", ["mpc_fail_steps", "mpc_fail_%", "emergency_stops", "planner_failures",
                "compute_ms_mean", "compute_ms_p95"]),
    ("Effort", ["mean_accel_cmd", "mean_speed", "path_length_per_uav_m"]),
]


def markdown(rows):
    lines = []
    for title, keys in GROUPS:
        lines += [f"## {title}", "", "| " + " | ".join(keys) + " |",
                  "|" + "---|" * len(keys)]
        lines += ["| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in rows]
        lines.append("")
    lines += ["## Leader sequence", ""]
    lines += [f"- **{r['scenario']}** ({r['run']}): {r['leader_sequence']}" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (default: latest run of each benchmark)")
    ap.add_argument("--out", default=os.path.join("results", "benchmark"),
                    help="output path without extension (writes .md and .csv)")
    a = ap.parse_args()

    runs = a.runs
    if not runs:
        runs = []
        for s in BENCHMARKS:
            r = latest_run(s)
            print(f"[EVAL] {s}: {r or 'no run found in runs/'}")
            if r:
                runs.append(r)
    if not runs:
        raise SystemExit("[EVAL] nothing to evaluate")

    rows = []
    for d in runs:
        print(f"[EVAL] scoring {d} ...")
        rows.append(evaluate(d))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    md = markdown(rows)
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("# Benchmark results\n\n" + md)
    keys = [k for _, ks in GROUPS for k in ks] + ["leader_sequence"]
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n" + md)
    print(f"[EVAL] wrote {a.out}.md and {a.out}.csv")


if __name__ == "__main__":
    main()
