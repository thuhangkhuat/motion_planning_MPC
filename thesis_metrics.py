"""
thesis_metrics.py — The four thesis metrics (metrics.py) on the benchmark runs.

    python thesis_metrics.py                    # latest run of each benchmark (scen1..scen4)
    python thesis_metrics.py --all              # every run of each benchmark: mean ± std over runs
    python thesis_metrics.py runs/<dir> ...     # given runs
    python thesis_metrics.py --all --per-run    # every run on its own line
    python thesis_metrics.py runs/<dir> --out results/<name>   # one run, its own output file
    python thesis_metrics.py --warmup 0         # count from step 0 (default: auto, see below)

The metrics are computed by metrics.run_metrics, the same code as
collect_metrics.py, but read from runs/<dir>/ (data.pkl + config.json):
  (1) Execution time      mean compute time per UAV per step [ms]
  (2) Total path length   sum over the UAVs [m] (and per UAV), and the part of
                          it flown in SEARCH, i.e. out of the formation: a
                          satellite is in SEARCH while it is farther from the
                          leader than the formation threshold (it enters TRACK
                          within VIEWING_RADIUS + FORMATION_GAP, leaves it
                          beyond that + TRACK_EXIT_HYSTERESIS); the leader is
                          always in TRACK. Needs the status recorded by main.py.
  (3) FOV coverage        union area of the FOVs connected to the leader [m²]
                          and efficiency = area / (N · (2L)²) ∈ [0, 1]
  (4) Tracking            visibility = % of steps with the target inside >= 1 FOV,
                          tracking error = mean distance target -> nearest UAV [m]

Collisions (same tests as evaluate.py: obstacle clearance < R, two UAVs
closer than 2R, outside the map). In the simulator a UAV flies on through an
obstacle, so what it does after its first collision has no physical meaning.
No run is dropped; instead each UAV counts as crashed from its first
collision on (as SPL / the CARLA driving score penalise a failure instead of
discarding the episode):
  Collision-free          1 if no UAV ever collides (mean over runs = success rate)
  First collision [s]     first collision of any UAV ("-" if none)
  UAV-time before coll.   % of UAV-steps before each UAV's first collision
  Safe visibility         visibility with the FOVs of the UAVs not yet crashed
  Safe coverage eff.      coverage efficiency with those FOVs, still divided by
                          the n FOVs of the full team
  Safe SEARCH path        SEARCH path of the UAVs not yet crashed; a crashed UAV
                          stops adding, so this is optimistic for runs with
                          collisions: read it with "UAV-time before coll."
  Mission visibility      strict version: the mission ends at the first
                          collision of any UAV (the convention of navigation
                          benchmarks, where a collision ends the episode); the
                          rest of the run counts as target not seen. The per-UAV
                          "Safe visibility" barely drops when only a satellite
                          crashes, since the leader keeps the target in view.
For a collision-free run the safe metrics equal the plain ones.

Warm-up: coverage and tracking skip the gathering at the start. "auto" = from
the first step where every UAV is in TRACK (needs the status recorded by
main.py); a number = that many seconds. Path length and execution time always
use the whole run.

Writes <out>.md (thesis table) and <out>.csv (one row per run).
"""

import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np

from evaluate import obstacle_clearance
from metrics import run_metrics

BENCHMARKS = ["scen1", "scen2", "scen3", "scen4"]

# (key, column title, format) in the thesis table
COLUMNS = [
    ("exec_ms", "(1) Exec. time [ms]", "{:.1f}"),
    ("total_path_len", "(2) Total path length [m]", "{:.0f}"),
    ("mean_path_len", "Path length / UAV [m]", "{:.0f}"),
    ("search_path_len", "Path length in SEARCH [m]", "{:.0f}"),
    ("fov_area", "(3) FOV coverage area [m²]", "{:.0f}"),
    ("fov_coverage_eff", "Coverage efficiency", "{:.3f}"),
    ("visibility", "(4) Visibility [%]", "{:.2f}"),
    ("track_err", "Tracking error [m]", "{:.2f}"),
    ("success", "Collision-free", "{:.2f}"),
    ("first_crash_s", "First collision [s]", "{:.1f}"),
    ("alive_uav_pct", "UAV-time before coll. [%]", "{:.1f}"),
    ("safe_visibility", "Safe visibility [%]", "{:.2f}"),
    ("safe_fov_coverage_eff", "Safe coverage eff.", "{:.3f}"),
    ("safe_search_path_len", "Safe SEARCH path [m]", "{:.0f}"),
    ("mission_visibility", "Mission visibility [%]", "{:.2f}"),
]


def runs_for(scenario, all_runs):
    runs = sorted(glob.glob(os.path.join("runs", f"{scenario}_*")), key=os.path.getmtime)
    runs = [r for r in runs if os.path.isfile(os.path.join(r, "data.pkl"))]
    return runs if all_runs else runs[-1:]


def warmup_steps(data, dt, warmup):
    if warmup != "auto":
        return int(round(float(warmup) / dt))
    ids = sorted(k for k in data if isinstance(k, int))
    st = [data[i].get("status") for i in ids]
    if any(s is None or len(s) == 0 for s in st):
        return 0
    T = min(len(s) for s in st)
    track = np.stack([np.asarray(s)[:T, 0] > 0 for s in st], 1).all(1)
    return int(np.argmax(track)) if track.any() else 0


def score(run_dir, warmup):
    with open(os.path.join(run_dir, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    sd, prm = cfg.get("scenario_def", {}), cfg.get("params", {})
    dt = float(prm.get("TIMESTEP", 0.1))
    vr = float(sd.get("viewing_radius", prm.get("VIEWING_RADIUS", 1.0)))
    fov_side = data.get("meta", {}).get("fov_side") or 2 * vr
    w = warmup_steps(data, dt, warmup)
    alive = alive_mask(data, sd, float(prm.get("ROBOT_RADIUS", 0.5)))
    m = run_metrics(data, fov_side=fov_side, robot_radius=float(prm.get("ROBOT_RADIUS", 0.5)),
                    timestep=dt, warmup_steps=w, alive=alive)
    m["exec_ms"] = 1e3 * m["exec_time"]
    m["search_path_len"] = search_path_length(data)
    m["safe_search_path_len"] = search_path_length(data, alive)
    first = np.flatnonzero(~alive.all(axis=1))
    m["success"] = float(not len(first))
    m["first_crash_s"] = float(first[0] * dt) if len(first) else float("nan")
    m["alive_uav_pct"] = 100.0 * float(alive.mean())
    if len(first):                                   # whole team "down" from the first collision
        team = np.repeat(alive.all(axis=1, keepdims=True), alive.shape[1], axis=1)
        m["mission_visibility"] = run_metrics(
            data, fov_side=fov_side, robot_radius=float(prm.get("ROBOT_RADIUS", 0.5)),
            timestep=dt, warmup_steps=w, alive=team)["safe_visibility"]
    else:
        m["mission_visibility"] = m["visibility"]
    m.update(run=os.path.basename(os.path.normpath(run_dir)),
             scenario=sd.get("name", data.get("meta", {}).get("scenario", "?")),
             uavs=len([k for k in data if isinstance(k, int)]), warmup_s=round(w * dt, 1))
    return m


def alive_mask(data, sd, R):
    """(T, n) bool: True until the UAV's first collision (obstacle clearance
    < R, another UAV closer than 2R, or outside the map), as in evaluate.py."""
    ids = sorted(k for k in data if isinstance(k, int))
    T = min(len(data[i]["path"]) for i in ids)
    P = np.stack([np.asarray(data[i]["path"])[:T, 1:3] for i in ids], 1)   # (T, n, 2)
    n = len(ids)
    hit = (obstacle_clearance(P.reshape(-1, 2), sd.get("rects") or [],
                              sd.get("circles") or []).reshape(T, n) < R)
    xlim, ylim = sd.get("xlim", [-np.inf, np.inf]), sd.get("ylim", [-np.inf, np.inf])
    hit |= ((P[..., 0] < xlim[0]) | (P[..., 0] > xlim[1])
            | (P[..., 1] < ylim[0]) | (P[..., 1] > ylim[1]))
    for a in range(n):
        for b in range(a + 1, n):
            close = np.linalg.norm(P[:, a] - P[:, b], axis=1) < 2 * R
            hit[:, a] |= close
            hit[:, b] |= close
    return ~(np.cumsum(hit, axis=0) > 0)


def search_path_length(data, alive=None):
    """Distance flown by all UAVs while in SEARCH (out of the formation), over
    the whole run; with `alive` (T, n), only before each UAV's first
    collision. nan for runs without the recorded status."""
    ids = sorted(k for k in data if isinstance(k, int))
    st = [data[i].get("status") for i in ids]
    if any(s is None or len(s) == 0 for s in st):
        return float("nan")
    T = min(min(len(data[i]["path"]) for i in ids), min(len(s) for s in st))
    total = 0.0
    for i, s in zip(ids, st):
        P = np.asarray(data[i]["path"])[:T, 1:3]
        step = np.linalg.norm(np.diff(P, axis=0), axis=1)        # step k: k -> k+1
        search = np.asarray(s)[1:T, 0] == 0                      # mode recorded after the step
        if alive is not None:
            search &= np.asarray(alive)[1:T, ids.index(i)]
        total += float(step[search].sum())
    return total


def cell(fmt, values):
    """One value, or mean ± std over runs; nan (e.g. no collision) is left
    out, "-" when every value is nan."""
    v = np.array(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "-"
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}"


def table(rows, per_run=False):
    """One line per scenario (mean ± std when a scenario has several runs),
    or one line per run with per_run=True."""
    if per_run:
        head = ["Scenario", "Run", "UAVs"] + [t for _, t, _ in COLUMNS]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in rows:
            cells = [r["scenario"], r["run"], str(r["uavs"])]
            cells += [cell(fmt, [r[k]]) for k, _, fmt in COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    head = ["Scenario", "UAVs", "Runs"] + [t for _, t, _ in COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == s]
        cells = [s, str(rs[0]["uavs"]), str(len(rs))]
        for k, _, fmt in COLUMNS:
            cells.append(cell(fmt, [r[k] for r in rs]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (default: the benchmarks in runs/)")
    ap.add_argument("--all", action="store_true",
                    help="use every run of each benchmark, not only the latest")
    ap.add_argument("--per-run", action="store_true",
                    help="one table line per run instead of mean ± std per scenario")
    ap.add_argument("--warmup", default="auto",
                    help="'auto' (all UAVs in TRACK) or seconds skipped for coverage/tracking")
    ap.add_argument("--out", default=os.path.join("results", "thesis_metrics"),
                    help="output path without extension (writes .md and .csv)")
    a = ap.parse_args()

    runs = a.runs or [r for s in BENCHMARKS for r in runs_for(s, a.all)]
    if not runs:
        raise SystemExit("[METRICS] no runs found (expected runs/scen1_* .. runs/scen4_*)")

    rows = []
    for d in runs:
        print(f"[METRICS] {d} ...")
        rows.append(score(d, a.warmup))

    md = ("# Thesis metrics\n\n" + table(rows, a.per_run) + "\n\n"
          "Warm-up skipped for coverage / tracking: "
          + ", ".join(f"{r['run']} {r['warmup_s']} s" for r in rows) + ".\n")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    keys = ["scenario", "run", "uavs", "warmup_s"] + [k for k, _, _ in COLUMNS] + [
        "control_energy", "min_inter_uav", "sim_time"]
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print("\n" + md)
    print(f"[METRICS] wrote {a.out}.md and {a.out}.csv")


if __name__ == "__main__":
    main()
