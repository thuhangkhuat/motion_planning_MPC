"""
collect_metrics.py — gom metrics từ các file bạn đã chạy tay
============================================================
Bạn chạy main.py nhiều lần (đổi NUMBER_RUN mỗi lần), mỗi lần ra 1 file:
    data{METHOD}_run_{NUMBER_RUN}_scen{SCENARIO}_{NUM_ROBOT}.txt

Script này KHÔNG chạy sim, chỉ đọc mọi file khớp pattern rồi tính metrics
cho từng run -> results_all.csv (mỗi dòng = 1 run) để plot_box.py vẽ.

Chạy:
    python collect_metrics.py --scenarios 1 2 7 8 --methods 1 2 3
    python collect_metrics.py --scenarios 7 --warmup 30 --cov-mode target_component
"""

import glob
import pickle
import argparse
import numpy as np

from metrics import run_metrics

METHOD_NAMES = {1: "APF", 2: "Purpose", 3: "MPC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, nargs="+", default=[7])
    ap.add_argument("--methods", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--warmup", type=int, default=0,
                    help="bỏ N bước đầu (giai đoạn tiếp cận) khi tính coverage/vis")
    ap.add_argument("--cov-mode", default="drop_isolated",
                    choices=["drop_isolated", "target_component", "largest"])
    ap.add_argument("--robot-radius", type=float, default=0.2)
    ap.add_argument("--timestep", type=float, default=0.1)
    ap.add_argument("--out", default="results_all.csv")
    args = ap.parse_args()

    header = ["scenario", "method", "method_name", "run",
              "exec_time", "total_path_len", "mean_path_len",
              "fov_area", "fov_coverage_eff", "visibility", "track_err",
              "control_energy", "min_inter_uav", "sim_time"]
    rows = []

    for s in args.scenarios:
        for m in args.methods:
            # data{m}_run_*_scen{s}_*.txt  (NUM_ROBOT không cần biết trước)
            pattern = f"data{m}_run_*_scen{s}_*.txt"
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"[SKIP] không thấy file nào khớp: {pattern}")
                continue
            for f in files:
                # tách NUMBER_RUN từ tên file
                try:
                    run_id = int(f.split("_run_")[1].split("_scen")[0])
                except (IndexError, ValueError):
                    run_id = -1
                with open(f, "rb") as fh:
                    data = pickle.load(fh)
                fov_side = data.get("meta", {}).get("fov_side")
                if fov_side is None:
                    print(f"[WARN] {f} thiếu meta.fov_side — bỏ qua")
                    continue
                mt = run_metrics(
                    data, fov_side=fov_side,
                    robot_radius=args.robot_radius, timestep=args.timestep,
                    cov_mode=args.cov_mode, warmup_steps=args.warmup)
                rows.append([s, m, METHOD_NAMES.get(m, str(m)), run_id,
                             mt["exec_time"], mt["total_path_len"],
                             mt["mean_path_len"], mt["fov_area"],
                             mt["fov_coverage_eff"], mt["visibility"],
                             mt["track_err"], mt["control_energy"],
                             mt["min_inter_uav"], mt["sim_time"]])
            print(f"[OK]   scen{s} {METHOD_NAMES.get(m, m):<12} "
                  f"{len(files)} run(s)")

    if not rows:
        print("Không có dữ liệu.")
        return

    with open(args.out, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[INFO] Saved {args.out} ({len(rows)} runs)")

    # summary nhanh mean±std
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r[0], r[2])].append(r)
    print("\n=== mean ± std qua các run ===")
    for (s, mn), rs in sorted(agg.items()):
        a = np.array([[x[4], x[5], x[8], x[9]] for x in rs], dtype=float)
        me, sd = a.mean(0), a.std(0)
        print(f"scen{s:>2} {mn:<12} "
              f"time={me[0]:.3f}±{sd[0]:.4f}s  "
              f"path={me[1]:.3f}±{sd[1]:.4f}m  "
              f"cov_eff={me[2]:.3f}±{sd[2]:.4f}  "
              f"vis={me[3]:.3f}±{sd[3]:.4f}%")


if __name__ == "__main__":
    main()