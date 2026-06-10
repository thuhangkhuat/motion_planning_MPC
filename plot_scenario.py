"""
plot_scenario.py — Tool xem trước kịch bản (obstacle + target path + UAV starts).

CÁCH DÙNG:
    # Vẽ scenario đang chọn trong config (SCENARIO):
    python plot_scenario.py

    # Vẽ scenario cụ thể:
    python plot_scenario.py 3

    # Vẽ kèm trajectory thực của target (chạy RRT + smooth, hơi lâu):
    python plot_scenario.py --traj
    python plot_scenario.py 3 --traj

HIỂN THỊ:
    - Obstacle chữ nhật/vuông (xám) + tròn (xám)
    - Waypoints target (đỏ, đánh số thứ tự) + đường nối (đứt nét)
    - Trajectory thực của target sau RRT + smoothing (nếu --traj)
    - Vị trí xuất phát UAV (tam giác xanh, đánh số)
    - Thông tin: density obstacle, số UAV, kích thước map
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_args():
    """Parse: [scenario_number] [--traj]"""
    scenario = None
    with_traj = False
    for a in sys.argv[1:]:
        if a == '--traj':
            with_traj = True
        else:
            try:
                scenario = int(a)
            except ValueError:
                print(f"Bỏ qua argument không hiểu: {a}")
    return scenario, with_traj


def main():
    scenario_arg, with_traj = parse_args()

    # QUAN TRỌNG: set env TRƯỚC khi import config để mọi module
    # (config, target_rrt, target_cache) cùng thấy đúng scenario —
    # obstacles, tốc độ target, cache key đều nhất quán.
    if scenario_arg is not None:
        import os
        os.environ["SCENARIO"] = str(scenario_arg)

    from config import (TAR_WAYPOINTS, STARTS, XLIM, YLIM,
                        SCENARIO, SCENARIOS)
    _s = SCENARIOS[SCENARIO]
    waypoints = TAR_WAYPOINTS
    starts = STARTS
    rects = _s['rects']
    circles = _s.get('circles', [])
    xlim, ylim = XLIM, YLIM
    scen_no = SCENARIO

    # ─── Tính density ───
    map_area = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])
    obs_area = sum(w * h for (_, _, w, h) in rects)
    obs_area += sum(np.pi * r * r for (_, _, r) in circles)
    density = obs_area / map_area * 100

    # ─── Plot ───
    w_fig = 9
    h_fig = w_fig * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, ax = plt.subplots(figsize=(w_fig, max(h_fig, 5)))

    # Obstacles
    for (x, y, w, h) in rects:
        ax.add_patch(patches.Rectangle((x, y), w, h,
                     facecolor='gray', alpha=0.7, edgecolor='black'))
    for (cx, cy, r) in circles:
        ax.add_patch(patches.Circle((cx, cy), r,
                     facecolor='gray', alpha=0.7, edgecolor='black'))

    # Waypoints + đường nối
    wp = np.array([w[:2] for w in waypoints])
    ax.plot(wp[:, 0], wp[:, 1], 'r--', linewidth=1.5, alpha=0.7,
            label='Waypoint polyline')
    ax.scatter(wp[:, 0], wp[:, 1], c='red', s=90, marker='o',
               zorder=5, edgecolors='darkred')
    for i, (x, y) in enumerate(wp):
        ax.annotate(f'W{i}', (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, color='darkred',
                    fontweight='bold')

    # Trajectory thực (optional)
    if with_traj:
        print("Generating target trajectory (RRT + smooth)...")
        try:
            from target_cache import load_or_generate_target
            target = load_or_generate_target(waypoints, verbose=True)
            traj = np.array([p[:2] for p in target.trajectory])
            ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.8,
                    label='Target trajectory (RRT + smooth)')
        except Exception as e:
            print(f"Không gen được trajectory: {e}")

    # UAV starts
    st = np.asarray(starts)
    ax.scatter(st[:, 0], st[:, 1], c='blue', s=130, marker='^',
               zorder=5, edgecolors='black', label='UAV starts')
    for i, (x, y) in enumerate(st[:, :2]):
        ax.annotate(f'U{i}', (x, y), textcoords="offset points",
                    xytext=(8, -12), fontsize=10, color='navy',
                    fontweight='bold')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    n_obs = len(rects) + len(circles)
    ax.set_title(f"Scenario {scen_no} — {st.shape[0]} UAV, "
                 f"{n_obs} obstacles, density {density:.1f}%")
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_png = f"scenario_{scen_no}_preview.png"
    plt.savefig(out_png, dpi=110)
    print(f"Saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()