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
  (2) Total path length   sum over the UAVs [m] (and per UAV)
  (3) FOV coverage        union area of the FOVs connected to the leader [m²]
                          and efficiency = area / (N · (2L)²) ∈ [0, 1]
  (4) Tracking            visibility = % of steps with the target inside >= 1 FOV,
                          tracking error = mean distance target -> nearest UAV [m]

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

from metrics import run_metrics

BENCHMARKS = ["scen1", "scen2", "scen3", "scen4"]

# (key, column title, format) in the thesis table
COLUMNS = [
    ("exec_ms", "(1) Exec. time [ms]", "{:.1f}"),
    ("total_path_len", "(2) Total path length [m]", "{:.0f}"),
    ("mean_path_len", "Path length / UAV [m]", "{:.0f}"),
    ("fov_area", "(3) FOV coverage area [m²]", "{:.0f}"),
    ("fov_coverage_eff", "Coverage efficiency", "{:.3f}"),
    ("visibility", "(4) Visibility [%]", "{:.2f}"),
    ("track_err", "Tracking error [m]", "{:.2f}"),
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
    m = run_metrics(data, fov_side=fov_side, robot_radius=float(prm.get("ROBOT_RADIUS", 0.5)),
                    timestep=dt, warmup_steps=w)
    m["exec_ms"] = 1e3 * m["exec_time"]
    m.update(run=os.path.basename(os.path.normpath(run_dir)),
             scenario=sd.get("name", data.get("meta", {}).get("scenario", "?")),
             uavs=len([k for k in data if isinstance(k, int)]), warmup_s=round(w * dt, 1))
    return m


def table(rows, per_run=False):
    """One line per scenario (mean ± std when a scenario has several runs),
    or one line per run with per_run=True."""
    if per_run:
        head = ["Scenario", "Run", "UAVs"] + [t for _, t, _ in COLUMNS]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in rows:
            cells = [r["scenario"], r["run"], str(r["uavs"])]
            cells += [fmt.format(float(r[k])) for k, _, fmt in COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    head = ["Scenario", "UAVs", "Runs"] + [t for _, t, _ in COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == s]
        cells = [s, str(rs[0]["uavs"]), str(len(rs))]
        for k, _, fmt in COLUMNS:
            v = np.array([r[k] for r in rs], float)
            cells.append(fmt.format(v.mean()) if len(v) == 1
                         else f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}")
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
