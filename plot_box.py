"""
plot_box.py — Box plots of the five evaluation metrics, one panel per metric,
one box per method, one sample per run (scenario x seed) from a thesis CSV.

    python plot_box.py results/thesis_10seeds.csv                       # one figure per scenario + all pooled
    python plot_box.py results/thesis_10seeds.csv --scenario scen3      # one scenario
    python plot_box.py results/thesis_10seeds.csv --methods ours impc_ob_rec cbf_qp_lag

Metrics: (1) exec. time mean [ms], (2a) gathering mean [s], (2b) re-formation
SEARCH [%], (3) mission coverage eff., (4) target loss [%].

Exec. time uses only the runs that did not overlap another run in time (from
the first / last time stamp of runs/<run>/log.txt): runs executed in parallel
share the CPU and their compute time is inflated. --exec-all keeps every run.
A run without a value (e.g. no UAV ever joined the formation) is left out of
that panel; the number of runs per box is printed.

Writes <out>_<scenario>.pdf / .png (default results/paper/box_<scenario>).
"""

import argparse
import csv
import datetime as dt
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
})

BOX_COLOR = "green"
MEDIAN_COLOR = "orange"

LABELS = {"ours": "Ours", "impc_ob_rec": "IMPC-OB", "cbf_qp_lag": "CBF-QP",
          "pure_mpc": "MPC", "apf": "APF", "apfc": "APF-C",
          "apfc_brake": "APF-C (brake)"}

# (CSV key, panel title)
METRICS = [
    ("exec_mean_ms", "Exec. time [ms]"),
    ("gather_time_mean_s", "Gathering time [s]"),
    ("reform_search_pct", "Re-formation SEARCH [%]"),
    ("mission_cov_eff", "Coverage efficiency"),
    ("target_loss_pct", "Target loss [%]"),
]

# paper numbering of the scenarios (same as plot_paper.PAPER_NAME)
PAPER_NAME = {"scen1": "scen1", "scen2": "scen3", "scen3": "scen6", "scen4": "scen2",
              "scen6": "scen4", "scen7": "scen5"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def run_interval(run_dir):
    lf = os.path.join(run_dir, "log.txt")
    if not os.path.isfile(lf):
        return None
    with open(lf, errors="ignore") as f:
        stamps = [l[:23] for l in f if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}", l)]
    if not stamps:
        return None
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    return dt.datetime.strptime(stamps[0], fmt), dt.datetime.strptime(stamps[-1], fmt)


def overlapping_runs(runs_root="runs"):
    """Names of the runs whose [start, end] overlaps any other run in runs_root."""
    iv = {}
    for d in glob.glob(os.path.join(runs_root, "*")):
        t = run_interval(d)
        if t:
            iv[os.path.basename(d)] = t
    bad = set()
    items = sorted(iv.items(), key=lambda kv: kv[1][0])
    for i, (a, (a0, a1)) in enumerate(items):
        for b, (b0, b1) in items[i + 1:]:
            if b0 >= a1:
                break
            bad.update((a, b))
    return bad


def plot(rows, methods, title, out, width, exec_bad):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(width, width * 0.26))
    for ax, (key, name) in zip(axes, METRICS):
        data, ns = [], []
        for m in methods:
            rs = [r for r in rows if r["controller"] == m]
            if key == "exec_mean_ms" and exec_bad is not None:
                rs = [r for r in rs if r["run"] not in exec_bad]
            if key == "target_loss_pct":
                v = [100.0 - num(r.get("mission_visibility")) if not r.get("target_loss_pct")
                     else num(r["target_loss_pct"]) for r in rs]
            else:
                v = [num(r.get(key)) for r in rs]
            v = np.asarray(v, float)
            v = v[~np.isnan(v)]
            data.append(v)
            ns.append(len(v))
        pos = [i + 1 for i, v in enumerate(data) if len(v)]
        ax.boxplot([v for v in data if len(v)], positions=pos, widths=0.5, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor=BOX_COLOR, edgecolor="black", linewidth=0.6),
                   medianprops=dict(color=MEDIAN_COLOR, linewidth=1.0),
                   whiskerprops=dict(color="black", linewidth=0.6),
                   capprops=dict(color="black", linewidth=0.6))
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([LABELS.get(m, m) for m in methods], rotation=45, ha="right")
        ax.set_xlim(0.4, len(methods) + 0.6)
        ax.set_title(name, pad=3)
        ax.yaxis.grid(True, color="0.75", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.5, length=2)
        print(f"   {name:26s} n = " + ", ".join(f"{LABELS.get(m, m)} {n}" for m, n in zip(methods, ns)))
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout(w_pad=0.8)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"[BOX] wrote {out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="per-run CSV written by thesis_metrics.py")
    ap.add_argument("--scenario", nargs="*", default=None,
                    help="scenarios (default: each scenario of the CSV, and all pooled)")
    ap.add_argument("--methods", nargs="+",
                    default=["ours", "impc_ob_rec", "cbf_qp_lag", "pure_mpc", "apf", "apfc"])
    ap.add_argument("--exec-all", action="store_true",
                    help="exec. time from every run, also the ones that overlapped in time")
    ap.add_argument("--runs-root", default="runs", help="where the run directories (log.txt) are")
    ap.add_argument("--width", type=float, default=7.16, help="figure width [in]")
    ap.add_argument("--title", action="store_true", help="scenario name above the figure")
    ap.add_argument("--out", default=os.path.join("results", "paper", "box"),
                    help="output prefix; _<scenario> is appended")
    a = ap.parse_args()

    with open(a.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["controller"] in a.methods]
    exec_bad = None if a.exec_all else overlapping_runs(a.runs_root)
    if exec_bad is not None:
        kept = sum(r["run"] not in exec_bad for r in rows)
        print(f"[BOX] exec. time from {kept}/{len(rows)} runs (no time overlap)")

    scens = a.scenario or list(dict.fromkeys(r["scenario"] for r in rows)) + ["all"]
    for s in scens:
        rs = rows if s == "all" else [r for r in rows if r["scenario"] == s]
        if not rs:
            print(f"[BOX] no rows for {s}")
            continue
        name = s if s == "all" else PAPER_NAME.get(s, s)
        plot(rs, a.methods, name if a.title else "", f"{a.out}_{name}", a.width, exec_bad)


if __name__ == "__main__":
    main()
