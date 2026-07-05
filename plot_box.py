
import argparse
import csv
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# metric -> (nhãn trục y, có phải "cao hơn tốt hơn" không — chỉ để chú thích)
METRICS = {
    "exec_time":        ("Computational time [s]", False),
    "total_path_len":   ("Total path length [m]",  False),
    "fov_coverage_eff": ("FOV coverage efficiency", True),
    "fov_area":         ("FOV coverage area [m$^2$]", True),
    "visibility":       ("Target visibility [%]",   True),
    "track_err":        ("Tracking error [m]",      False),
}

# thứ tự + màu các method (giống style hình: xanh lá / xanh dương / vàng)
METHOD_ORDER = ["APF", "PureMPC", "JPS+MPC+CBF"]
METHOD_COLORS = {
    "APF":         "#f4d03f",   # vàng
    "PureMPC":     "#5dade2",   # xanh dương
    "JPS+MPC+CBF": "#58d68d",   # xanh lá (đề xuất)
}


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def make_figure(rows, metric, ylabel, outfile):
    scenarios = sorted(set(int(r["scenario"]) for r in rows))
    methods = [m for m in METHOD_ORDER
               if any(r["method_name"] == m for r in rows)]

    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    axes = axes[0]

    for col, s in enumerate(scenarios):
        ax = axes[col]
        data, colors = [], []
        for m in methods:
            vals = [float(r[metric]) for r in rows
                    if int(r["scenario"]) == s and r["method_name"] == m
                    and r[metric] not in ("", "nan")]
            data.append(vals)
            colors.append(METHOD_COLORS.get(m, "#cccccc"))

        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.4),
                        whiskerprops=dict(color="black"),
                        capprops=dict(color="black"),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="gray", alpha=0.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor("black")
            patch.set_alpha(0.9)

        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=15, fontsize=8)
        ax.grid(axis="y", ls="--", alpha=0.5)
        ax.set_title(f"({chr(97 + col)}) Scenario {s}", fontsize=10)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_all.csv")
    ap.add_argument("--metrics", nargs="+",
                    default=["exec_time", "total_path_len",
                             "fov_coverage_eff", "visibility"])
    args = ap.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        print("CSV rỗng.")
        return

    for metric in args.metrics:
        if metric not in METRICS:
            print(f"[SKIP] metric không biết: {metric}")
            continue
        ylabel, _ = METRICS[metric]
        make_figure(rows, metric, ylabel, f"box_{metric}.png")


if __name__ == "__main__":
    main()