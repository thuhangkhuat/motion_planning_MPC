from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

from matplotlib.patches import Circle, Rectangle
from config import *


# ============================================================
# TÙY CHỈNH
# ============================================================
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
LEADER_COLOR = "#f1c40f"
FOV_FOLLOWS_DRONE = True     # True: FOV cùng màu drone | False: FOV một màu
FOV_COLOR = "#2ca02c"
TRAIL_ALPHA = 0.9            # độ đậm đường đã đi
FRAME_STEP = 1              # vẽ mỗi N frame (tăng để GIF nhẹ/nhanh hơn)

export = True
if export:
    import cv2
    image_array = []
STANDARD_SIZE = (1280, 720)


# ============================================================
# MARKER HELPERS (đồng bộ với plot_scenario.py)
# ============================================================
def getCircle(x, y, r):
    theta = np.linspace(0, 2 * np.pi, 150)
    return x + r * np.cos(theta), y + r * np.sin(theta)


def draw_drone(ax, x, y, yaw=0.0, color="#1f77b4", is_leader=False,
               zorder=10, show_radius=True):
    """
    Icon drone (chữ thập X) vẽ ĐÚNG kích thước vật lý theo ROBOT_RADIUS.
    - Cánh dài = ROBOT_RADIUS  -> chữ thập nằm gọn trong bán kính thật.
    - show_radius=True: vẽ thêm vòng tròn bán kính ROBOT_RADIUS (vùng va chạm thật).
    """
    r = ROBOT_RADIUS
    d = r * 0.9                      # nửa chiều dài cánh ~ trong bán kính
    for a in (np.pi / 4, 3 * np.pi / 4):
        ax.plot([x - d * np.cos(a + yaw), x + d * np.cos(a + yaw)],
                [y - d * np.sin(a + yaw), y + d * np.sin(a + yaw)],
                color=color, lw=1.4, zorder=zorder, solid_capstyle="round")
    ax.plot(x, y, "o", ms=3, mfc="#222", mec=color, mew=0.8, zorder=zorder + 1)
    if show_radius:
        ax.add_patch(Circle((x, y), r, fc="none", ec=color, lw=0.8,
                            alpha=0.6, zorder=zorder - 1))
    if is_leader:
        ax.add_patch(Circle((x, y), r * 1.8, fc="none", ec=LEADER_COLOR,
                            lw=1.6, zorder=zorder - 1, alpha=0.95))


def draw_target(ax, x, y, scale=1.0, zorder=12, label=None):
    ax.plot(x, y, marker="*", ms=16 * scale, mfc="#e11", mec="#7a0000",
            mew=1.3, zorder=zorder, label=label, linestyle="None")


def draw_fov_square(ax, x, y, L, color, alpha=0.15, edge_alpha=0.45, zorder=2):
    ax.add_patch(Rectangle((x - L, y - L), 2 * L, 2 * L, fc=color,
                          ec=color, lw=0.6, alpha=alpha, zorder=zorder))
    if edge_alpha > 0:
        ax.add_patch(Rectangle((x - L, y - L), 2 * L, 2 * L, fc="none",
                              ec=color, lw=0.8, alpha=edge_alpha, zorder=zorder))


def plot_convex_polygon(ax, A, b, color):
    if (A is None or b is None or not hasattr(A, '__len__')
            or not hasattr(b, '__len__') or len(A) == 0 or len(b) == 0):
        return
    if isinstance(A, list) and len(A) == 1:
        A = A[0]
    if isinstance(b, list) and len(b) == 1:
        b = b[0]
    try:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
    except ValueError:
        return
    if A.size == 0 or b.size == 0:
        return
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_grid, y_grid = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                                 np.linspace(ylim[0], ylim[1], 100))
    pts = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    inside = np.all(pts @ A.T - b.flatten() <= 1e-5, axis=1)
    ax.scatter(pts[inside, 0], pts[inside, 1], color=color, alpha=0.3,
               s=7, ec='none')


# ============================================================
# LOAD
# ============================================================
with open(FILE_NAME, 'rb') as file:
    data = pickle.load(file)
target_trajectory = data[0]["tar_traj"]

sizes = {1: (12, 5.5), 2: (12, 5.5), 3: (12, 5.5), 4: (12, 5.5),
         5: (12, 5.5), 6: (12, 5.5), 7: (12, 5.5), 8: (12, 5.5)}
size = sizes.get(SCENARIO, (12, 5.5))
plt.figure(figsize=size)
length = min(data[0]["path"].shape[0], target_trajectory.shape[0])
ax = plt.axes()


def leader_at(iter):
    """UAV gần target nhất (Chebyshev) ở frame iter -> leader."""
    tx, ty = target_trajectory[iter, 0], target_trajectory[iter, 1]
    d = [max(abs(data[i]["path"][iter, 1] - tx),
             abs(data[i]["path"][iter, 2] - ty)) for i in range(NUM_ROBOT)]
    return int(np.argmin(d))


for iter in range(0, length, FRAME_STEP):
    ax.cla()

    # start
    ax.scatter(STARTS[:, 0], STARTS[:, 1], marker="s", s=50, color="#333")

    # obstacles
    for j in range(OBSTACLES.shape[0]):
        x, y, r = OBSTACLES[j, :]
        a, b = getCircle(x, y, r)
        ax.fill(a, b, color="black")
        ax.plot(a, b, color='k', lw=1.5)
    for poly in POLYGON_OBSTACLES:
        ax.fill(poly[:, 0], poly[:, 1], color='black')
        ax.plot(np.append(poly[:, 0], poly[0, 0]),
                np.append(poly[:, 1], poly[0, 1]), 'k-', lw=1.5)

    # target: vệt đã đi (mờ) + ngôi sao hiện tại
    ax.plot(target_trajectory[:iter + 1, 0], target_trajectory[:iter + 1, 1],
            'r--', lw=1.0, alpha=0.5)
    draw_target(ax, target_trajectory[iter, 0], target_trajectory[iter, 1])

    lead = leader_at(iter)

    for i in range(NUM_ROBOT):
        color = COLORS[i % len(COLORS)]
        fov_color = color if FOV_FOLLOWS_DRONE else FOV_COLOR
        path = data[i]["path"]
        corridors_data = data[i]["corridors"]

        pos = path[iter, 1:4]

        # FOV vuông của UAV ở frame hiện tại
        draw_fov_square(ax, pos[0], pos[1], VIEWING_RADIUS, fov_color)

        # đường đã đi tới frame này
        ax.plot(path[:iter + 1, 1], path[:iter + 1, 2], color=color,
                lw=1.3, alpha=TRAIL_ALPHA)

        # corridor hiện tại (bật nếu muốn)
        # if iter < len(corridors_data):
        #     c = corridors_data[iter]
        #     plot_convex_polygon(ax, c.get('A'), c.get('b'), color)

        # icon drone — GIỮ CỐ ĐỊNH, vẽ đúng kích thước ROBOT_RADIUS
        draw_drone(ax, pos[0], pos[1], yaw=0.0, color=color,
                   is_leader=(i == lead))

    ax.axis("scaled")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    plt.tight_layout()

    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    if export:
        file_name = "results/data.png"
        plt.savefig(file_name)
        img = cv2.imread(file_name)
        image_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.pause(0.001)

if export:
    import imageio
    imageio.mimsave(SAVE_GIF, image_array)
    print(f"Saved {SAVE_GIF}")

plt.show()