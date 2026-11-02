"""
thesis_metrics.py — The four thesis metrics (metrics.py) on the benchmark runs.

    python thesis_metrics.py                    # latest run of each benchmark (scen1..scen4)
    python thesis_metrics.py --all              # every run of each benchmark: mean ± std over runs
    python thesis_metrics.py runs/<dir> ...     # given runs
    python thesis_metrics.py --all --per-run    # every run on its own line
    python thesis_metrics.py runs/<dir> --out results/<name>   # one run, its own output file
    python thesis_metrics.py --warmup 0         # count from step 0 (default: auto, see below)

Collisions (same tests as evaluate.py: obstacle clearance < R, two UAVs
closer than 2R, outside the map): in the simulator a UAV flies on through an
obstacle, so nothing after the team's first collision t_c has a physical
meaning. No run is dropped; the metrics are cut at t_c (as SPL / the CARLA
driving score penalise a failure instead of discarding the episode):
  Collision-free          1 if no UAV ever collides (mean over runs = success rate)
  Completion [%]          t_c / T (100 without a collision)
  (1) Exec. time          compute time per UAV per step before t_c [ms]: mean,
                          95th percentile and max (the control period is TIMESTEP)
  (2a) Gathering SEARCH   path flown by each UAV from the start until it first
                          enters TRACK (joins the formation; the leader is in TRACK
                          from its election), summed over the UAVs [m]; "-" if a UAV
                          has not joined before t_c (gathered_uavs in the CSV)
  (2b) Re-formation       SEARCH path after each UAV first joined the formation
       SEARCH             and before t_c, per km of target path, for the whole
                          team [m / km]: N * sum_i S_i / sum_i D_i, with S_i the
                          SEARCH path of UAV i and D_i the target km over the same
                          window; "-" if the mean window is < 20 % of the mission
  (3) Mission coverage    coverage efficiency (FOV union connected to the leader /
                          (N (2L)^2)) with every FOV dropped from t_c on, i.e.
                          coverage x completion
  (4) Mission visibility  % of steps with the target inside >= 1 FOV, the steps
                          from t_c on counting as not seen, i.e. visibility x completion
SEARCH (out of the formation): a satellite farther from the leader than the
formation threshold (enters TRACK within VIEWING_RADIUS + FORMATION_GAP, leaves
it beyond that + TRACK_EXIT_HYSTERESIS), or every UAV while no UAV sees the
target. Needs the status recorded by main.py.

Warm-up: (3) and (4) skip the gathering at the start. "auto" = from the first
step where every UAV is in TRACK; a number = that many seconds.

The CSV also has the plain metrics (whole run, collisions ignored): total and
SEARCH path length, coverage efficiency, visibility, mean exec time.

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
    ("success", "Collision-free", "{:.2f}"),
    ("completion_pct", "Completion [%]", "{:.1f}"),
    ("exec_mean_ms", "(1) Exec. mean [ms]", "{:.1f}"),
    ("exec_p95_ms", "Exec. p95 [ms]", "{:.1f}"),
    ("exec_max_ms", "Exec. max [ms]", "{:.0f}"),
    ("gather_search_m", "(2a) Gathering SEARCH [m]", "{:.0f}"),
    ("reform_search_m_per_km", "(2b) Re-formation SEARCH [m/km]", "{:.0f}"),
    ("mission_cov_eff", "(3) Mission coverage eff.", "{:.3f}"),
    ("mission_visibility", "(4) Mission visibility [%]", "{:.2f}"),
]
# plain metrics (whole run, collisions ignored), in the CSV only
EXTRA = ["gathered_uavs", "first_crash_s", "total_path_len", "search_path_len",
         "fov_coverage_eff", "visibility", "exec_ms", "warmup_s", "sim_time"]
REFORM_MIN_WINDOW = 0.2        # (2b) needs a mean window of at least this fraction of the run


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
    R = float(prm.get("ROBOT_RADIUS", 0.5))
    alive = alive_mask(data, sd, R)
    T = alive.shape[0]
    first = np.flatnonzero(~alive.all(axis=1))
    kc = int(first[0]) if len(first) else T                  # first collision of the team
    m = run_metrics(data, fov_side=fov_side, robot_radius=R, timestep=dt, warmup_steps=w)
    m["exec_ms"] = 1e3 * m["exec_time"]
    m["search_path_len"] = search_path_length(data)
    m["success"] = float(kc == T)
    m["first_crash_s"] = float(kc * dt) if kc < T else float("nan")
    m["completion_pct"] = 100.0 * kc / T
    # (3), (4): every FOV dropped from the first collision on
    if kc < T:
        team = np.repeat((np.arange(T) < kc)[:, None], alive.shape[1], axis=1)
        mm = run_metrics(data, fov_side=fov_side, robot_radius=R, timestep=dt,
                         warmup_steps=w, alive=team)
        m["mission_visibility"], m["mission_cov_eff"] = mm["safe_visibility"], mm["safe_fov_coverage_eff"]
    else:
        m["mission_visibility"], m["mission_cov_eff"] = m["visibility"], m["fov_coverage_eff"]
    m.update(exec_stats(data, kc))
    m.update(search_split(data, kc))
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


def search_path_length(data):
    """Distance flown by all UAVs while in SEARCH (out of the formation), over
    the whole run; nan for runs without the recorded status."""
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
        total += float(step[search].sum())
    return total


def exec_stats(data, kc):
    """(1): compute time per UAV per step [ms] before the step kc (main.py
    records one time per UAV and step, the UAVs of a step one after another)."""
    ct = 1e3 * np.asarray(data.get("meta", {}).get("compute_times", []), float)
    n = len([k for k in data if isinstance(k, int)])
    if not ct.size or not n:
        return dict(exec_mean_ms=np.nan, exec_p95_ms=np.nan, exec_max_ms=np.nan)
    ct = ct[:(len(ct) // n) * n].reshape(-1, n)[:max(kc, 1)].ravel()
    return dict(exec_mean_ms=float(ct.mean()), exec_p95_ms=float(np.percentile(ct, 95)),
                exec_max_ms=float(ct.max()))


def search_split(data, kc):
    """(2a) gathering and (2b) re-formation SEARCH path, each UAV split at its
    own first entry into TRACK, both cut at the team's first collision kc."""
    ids = sorted(k for k in data if isinstance(k, int))
    st = [data[i].get("status") for i in ids]
    nan = dict(gather_search_m=np.nan, reform_search_m_per_km=np.nan, gathered_uavs="-")
    if any(s is None or len(s) == 0 for s in st):
        return nan
    T = min(min(len(data[i]["path"]) for i in ids), min(len(s) for s in st))
    kc = min(kc, T)
    tar = np.asarray(data[ids[0]]["tar_traj"])[:T, :2]
    tstep = np.linalg.norm(np.diff(tar, axis=0), axis=1)          # target, step k: k -> k+1
    gather, S, D, win, joined = 0.0, 0.0, 0.0, 0, 0
    for i, s in zip(ids, st):
        P = np.asarray(data[i]["path"])[:T, 1:3]
        step = np.linalg.norm(np.diff(P, axis=0), axis=1)
        track = np.asarray(s)[:T, 0] > 0
        if not track[:kc].any():
            continue                                             # never joined before kc
        kt = int(np.argmax(track))
        joined += 1
        gather += float(step[:kt].sum())
        search = ~track[kt + 1:kc]                               # mode after the steps kt .. kc-2
        S += float(step[kt:kc - 1][search].sum())
        D += float(tstep[kt:kc - 1].sum()) / 1000.0
        win += kc - kt
    n = len(ids)
    out = dict(gathered_uavs=f"{joined}/{n}",
               gather_search_m=gather if joined == n else np.nan,
               reform_search_m_per_km=np.nan)
    if joined and D > 0 and win / joined >= REFORM_MIN_WINDOW * T:
        out["reform_search_m_per_km"] = n * S / D
    return out


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
    keys = ["scenario", "run", "uavs"] + [k for k, _, _ in COLUMNS] + EXTRA
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print("\n" + md)
    print(f"[METRICS] wrote {a.out}.md and {a.out}.csv")


if __name__ == "__main__":
    main()
