"""
stats_seeds.py — Paired comparison of "ours" against each baseline over the
(scenario, seed) pairs of thesis_metrics.py CSVs (Wilcoxon signed-rank test).

    python stats_seeds.py results/thesis_seeds.csv
    python stats_seeds.py results/thesis_scen*.csv --ref ours --out results/stats_seeds

A pair is the same scenario and seed (so the same jittered UAV starts) for both
controllers. A metric that is "-" for a run (crashed too early) drops that pair;
n gives the pairs kept. Completion counts every pair. For each metric: the
pairs where ours is better / worse, the median of (ours - baseline) and the
two-sided p-value (exact for n <= 25, no p for n < 6 or all differences zero).

Writes <out>.md and <out>.csv.
"""

import argparse
import csv
import os

import numpy as np
from scipy.stats import wilcoxon

# (CSV key, title, +1 higher is better / -1 lower is better)
METRICS = [
    ("completion_pct", "Completion [%]", +1),
    ("exec_mean_ms", "(1) Exec. mean [ms]", -1),
    ("exec_p95_ms", "Exec. p95 [ms]", -1),
    ("exec_max_ms", "Exec. max [ms]", -1),
    ("gather_time_mean_s", "(2a) Gathering mean [s]", -1),
    ("gather_time_team_s", "Gathering team [s]", -1),
    ("reform_search_pct", "(2b) Re-formation SEARCH [%]", -1),
    ("mission_cov_eff", "(3) Mission coverage eff.", +1),
    ("mission_visibility", "(4) Mission visibility [%]", +1),
]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    runs = {}
    for r in rows:
        if not r.get("controller"):
            r["controller"] = r["run"].rsplit("_s", 1)[-1].split("_", 2)[-1]
        runs[(r["controller"], r["scenario"], r.get("seed", ""))] = r   # duplicates: last one wins
    return runs


def compare(runs, ref, base):
    keys = sorted(k[1:] for k in runs if k[0] == ref and (base,) + k[1:] in runs)
    out = []
    for key, title, sign in METRICS:
        a = np.array([num(runs[(ref,) + k][key]) for k in keys])
        b = np.array([num(runs[(base,) + k][key]) for k in keys])
        ok = np.isfinite(a) & np.isfinite(b)
        d = sign * (a[ok] - b[ok])                          # > 0: ours better
        p = np.nan
        if ok.sum() >= 6 and np.any(d != 0):
            p = float(wilcoxon(d).pvalue)
        out.append(dict(baseline=base, metric=title, n=int(ok.sum()), pairs=len(keys),
                        better=int((d > 0).sum()), worse=int((d < 0).sum()),
                        median_diff=float(np.median(a[ok] - b[ok])) if ok.any() else np.nan,
                        p=p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="thesis_metrics.py CSV file(s)")
    ap.add_argument("--ref", default="ours", help="controller compared against the others")
    ap.add_argument("--out", default=os.path.join("results", "stats_seeds"))
    a = ap.parse_args()

    runs = load(a.csv)
    bases = sorted({k[0] for k in runs} - {a.ref})
    res = [r for b in bases for r in compare(runs, a.ref, b)]

    def fmt_p(p):
        return "-" if not np.isfinite(p) else ("<0.001" if p < 1e-3 else f"{p:.3f}")
    lines = [f"# {a.ref} vs baselines: paired Wilcoxon over (scenario, seed)\n",
             "| Baseline | Metric | n | Better | Worse | Median (ours - base) | p |",
             "|---|---|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['baseline']} | {r['metric']} | {r['n']}/{r['pairs']} | {r['better']} | "
                     f"{r['worse']} | {r['median_diff']:.3g} | {fmt_p(r['p'])} |")
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(res[0]))
        wr.writeheader()
        wr.writerows(res)
    print(md)
    print(f"[STATS] wrote {a.out}.md and {a.out}.csv")


if __name__ == "__main__":
    main()
