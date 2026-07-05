from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle

from matplotlib.patches import Circle, Rectangle
from config import *


# ============================================================
# TÙY CHỈNH HIỂN THỊ
# ============================================================
# Mỗi drone một màu (path/icon/FOV lấy theo COLORS[i])
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', "#a4a426", "#16b9cb"]
LEADER_COLOR = "#f1c40f"
FOV_FOLLOWS_DRONE = True     # True: FOV cùng màu drone | False: FOV một màu chung
FOV_COLOR = "#2ca02c"        # dùng khi FOV_FOLLOWS_DRONE = False

FOV_SPACING = 2          # m — rải 1 ô FOV mỗi 0.75m đường đi (nhỏ = dày hơn)
FOV_ALPHA   = 0.15         # độ mờ mỗi ô FOV (nhỏ = chồng nhiều không quá đậm)

SHOW_2D = True              # vẽ hình 2D (top-down)
SHOW_3D = True              # vẽ thêm hình 3D
OBS_HEIGHT = 4.0            # chiều cao đùn obstacle trong view 3D (m)
Z_FALLBACK = 3.0            # độ cao UAV nếu path không có cột z hợp lệ
HIDE_3D_TICK_NUMBERS = True # ẩn SỐ trên trục 3D (vẫn giữ khung/lưới)

# ─── Khắc phục 3D bị nhỏ so với khoảng trắng ───
FIG3D_SIZE = (8, 1.0)      # figure thấp-rộng cho khớp map dài
Z_ASPECT = 1.0             # phóng đại trục z (khối dày hơn; 1.0 = tỉ lệ thật)
ZOOM_3D = 1.2            # >1 phóng to hình trong khung (bớt khoảng trắng)
ELEV_3D, AZIM_3D = 28, -62 # góc nhìn 3D

export = True


# ============================================================
# HELPERS 2D
# ============================================================
def getCircle(x, y, r):
    theta = np.linspace(0, 2 * np.pi, 150)
    return x + r * np.cos(theta), y + r * np.sin(theta)


def draw_drone(ax, x, y, yaw=0.0, scale=1.0, color="#1f77b4",
               is_leader=False, zorder=10):
    """Quadrotor tối giản: 2 nét chéo (chữ thập X) + tâm. Leader -> vòng vàng."""
    d = 0.6 * scale
    for a in (np.pi / 4, 3 * np.pi / 4):
        ax.plot([x - d * np.cos(a + yaw), x + d * np.cos(a + yaw)],
                [y - d * np.sin(a + yaw), y + d * np.sin(a + yaw)],
                color=color, lw=2.2, zorder=zorder, solid_capstyle="round")
    ax.plot(x, y, "o", ms=4 * scale, mfc="#222", mec=color, mew=1.0,
            zorder=zorder + 1)
    # if is_leader:
    #     ax.add_patch(Circle((x, y), 1.1 * scale, fc="none", ec=LEADER_COLOR,
    #                         lw=2.0, zorder=zorder - 1, alpha=0.95))


def draw_target(ax, x, y, scale=1.0, zorder=12, label=None):
    """Marker target: ngôi sao đỏ — khác hẳn dấu X của drone."""
    ax.plot(x, y, marker="*", ms=16 * scale, mfc="#e11", mec="#7a0000",
            mew=1.3, zorder=zorder, label=label, linestyle="None")


def draw_fov_square(ax, x, y, L, color, alpha=0.12, edge_alpha=0.5, zorder=2):
    """FOV vuông trục-chuẩn cạnh 2L tâm (x,y) — KHỚP định nghĩa metric."""
    ax.add_patch(Rectangle((x - L, y - L), 2 * L, 2 * L, fc=color,
                          ec=color, lw=0.6, alpha=alpha, zorder=zorder))
    if edge_alpha > 0:
        ax.add_patch(Rectangle((x - L, y - L), 2 * L, 2 * L, fc="none",
                              ec=color, lw=0.8, alpha=edge_alpha, zorder=zorder))


def draw_fov_along_path(ax, path_xy, L, color, spacing=FOV_SPACING,
                        alpha=FOV_ALPHA):
    """Rải FOV đều theo QUÃNG ĐƯỜNG -> dải FOV liền mạch. Ô cuối tô đậm hơn."""
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) < 2:
        return
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))]
    total = d[-1]
    if total <= 0:
        return
    s = np.arange(0.0, total, max(spacing, 1e-3))
    xs = np.interp(s, d, path_xy[:, 0])
    ys = np.interp(s, d, path_xy[:, 1])
    for x, y in zip(xs, ys):
        draw_fov_square(ax, x, y, L, color, alpha=alpha, edge_alpha=0.0)
    draw_fov_square(ax, path_xy[-1, 0], path_xy[-1, 1], L, color,
                    alpha=0.18, edge_alpha=0.5)


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
# HELPERS 3D
# ============================================================
def _box_faces(x, y, w, h, z0, z1):
    c = [(x, y, z0), (x + w, y, z0), (x + w, y + h, z0), (x, y + h, z0),
         (x, y, z1), (x + w, y, z1), (x + w, y + h, z1), (x, y + h, z1)]
    return [[c[0], c[1], c[2], c[3]], [c[4], c[5], c[6], c[7]],
            [c[0], c[1], c[5], c[4]], [c[2], c[3], c[7], c[6]],
            [c[1], c[2], c[6], c[5]], [c[0], c[3], c[7], c[4]]]


def _uav_z(path, idx):
    """Lấy độ cao z của UAV tại idx (cột 3 của path); fallback nếu ~0."""
    z = path[idx, 3]
    return z if abs(z) > 1e-6 else Z_FALLBACK


# ============================================================
# LOAD DATA
# ============================================================
with open(FILE_NAME, 'rb') as file:
    data = pickle.load(file)
target_trajectory = data[0]["tar_traj"]
length = min(data[0]["path"].shape[0], target_trajectory.shape[0])
iter_final = length - 1

# leader ở frame cuối: UAV gần target nhất (Chebyshev)
tgt_xy = target_trajectory[iter_final, :2]
d_inf = [max(abs(data[i]["path"][iter_final, 1] - tgt_xy[0]),
             abs(data[i]["path"][iter_final, 2] - tgt_xy[1]))
         for i in range(NUM_ROBOT)]
leader_index_at_final = int(np.argmin(d_inf))


# ============================================================
# 2D TOP-DOWN
# ============================================================
if SHOW_2D:
    size = (12, 7.5)
    fig = plt.figure(figsize=size)
    ax = plt.axes()

    ax.scatter(STARTS[:, 0], STARTS[:, 1], marker="s", s=50, color="#030c70", zorder=6)

    for j in range(OBSTACLES.shape[0]):
        x, y, r = OBSTACLES[j, :]
        a, b = getCircle(x, y, r)
        ax.fill(a, b, color="black")
        ax.plot(a, b, color='k', lw=1.5)
    for poly in POLYGON_OBSTACLES:
        ax.fill(poly[:, 0], poly[:, 1], color='black',
                label="Obstacle" if (OBSTACLES.shape[0] == 0
                                     and poly is POLYGON_OBSTACLES[0]) else "")
        ax.plot(np.append(poly[:, 0], poly[0, 0]),
                np.append(poly[:, 1], poly[0, 1]), 'k-', lw=1.5)

    ax.plot(target_trajectory[:length, 0], target_trajectory[:length, 1],
            'r--', lw=1.2, alpha=0.6, label="Target path")
    draw_target(ax, target_trajectory[iter_final, 0],
                target_trajectory[iter_final, 1], label="Target")

    for i in range(NUM_ROBOT):
        color = COLORS[i % len(COLORS)]
        fov_color = color if FOV_FOLLOWS_DRONE else FOV_COLOR
        path = data[i]["path"]
        corridors_data = data[i]["corridors"]

        ax.plot(path[:, 1], path[:, 2], color=color, lw=1.3,
                label=f"Drone {i+1}")
        draw_fov_along_path(ax, path[:length, 1:3], VIEWING_RADIUS, fov_color)

        pos = path[iter_final, 1:4]
        vel = path[iter_final, 4:6]
        yaw = (math.atan2(vel[1], vel[0])
               if np.linalg.norm(vel) > 1e-5 else 0.0)
        draw_drone(ax, pos[0], pos[1], yaw, scale=1.0, color=color,
                   is_leader=(i == leader_index_at_final))

        # if iter_final < len(corridors_data):
        #     corridor = corridors_data[iter_final]
        #     plot_convex_polygon(ax, corridor.get('A'), corridor.get('b'),
        #                         color)

    # legend entries đại diện (không lặp)
    # ax.plot([], [], 'o', mfc="none", mec=LEADER_COLOR, mew=2, label="Leader")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axis("scaled")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    plt.tight_layout()
    if export:
        plt.savefig("results/final_figure.png", dpi=300)
        print("Saved results/final_figure.png")


# ============================================================
# 3D VIEW
# ============================================================
if SHOW_3D:
    fig3d = plt.figure(figsize=FIG3D_SIZE)
    ax3 = fig3d.add_subplot(111, projection='3d')

    # obstacles (đùn khối)
    for rc in RECTANGLE_OBSTACLES:
        x, y, w, h = float(rc[0]), float(rc[1]), float(rc[2]), float(rc[3])
        ax3.add_collection3d(Poly3DCollection(
            _box_faces(x, y, w, h, 0, OBS_HEIGHT),
            facecolor="#555", edgecolor="#333", alpha=0.5, linewidths=0.3))
    for j in range(OBSTACLES.shape[0]):
        cx, cy, r = OBSTACLES[j, :]
        th = np.linspace(0, 2 * np.pi, 24)
        z = np.linspace(0, OBS_HEIGHT, 2)
        T, Z = np.meshgrid(th, z)
        ax3.plot_surface(cx + r * np.cos(T), cy + r * np.sin(T), Z,
                         color="#555", alpha=0.5, linewidth=0)

    # target path (ground) + star
    ax3.plot(target_trajectory[:length, 0], target_trajectory[:length, 1], 0,
             'r--', lw=1.3, alpha=0.7, label="Target path")
    ax3.scatter(target_trajectory[iter_final, 0],
                target_trajectory[iter_final, 1], 0, marker="*", s=260,
                c="#e11", edgecolor="#7a0000", depthshade=False,
                label="Target", zorder=20)

    for i in range(NUM_ROBOT):
        color = COLORS[i % len(COLORS)]
        fov_color = color if FOV_FOLLOWS_DRONE else FOV_COLOR
        path = data[i]["path"]
        zcol = np.where(np.abs(path[:length, 3]) > 1e-6, path[:length, 3],
                        Z_FALLBACK)
        ax3.plot(path[:length, 1], path[:length, 2], zcol, color=color,
                 lw=1.4, label=f"Drone {i}")

        # FOV footprint + frustum ở frame cuối
        x, y = path[iter_final, 1], path[iter_final, 2]
        z = _uav_z(path, iter_final)
        L = VIEWING_RADIUS
        quad = [[(x - L, y - L, 0), (x + L, y - L, 0),
                 (x + L, y + L, 0), (x - L, y + L, 0)]]
        ax3.add_collection3d(Poly3DCollection(
            quad, facecolor=fov_color, alpha=0.22, edgecolor=fov_color,
            linewidths=1.0))
        for cx, cy in [(x - L, y - L), (x + L, y - L),
                       (x + L, y + L), (x - L, y + L)]:
            ax3.plot([x, cx], [y, cy], [z, 0], color=fov_color, lw=0.6,
                     alpha=0.5)
        ax3.scatter(x, y, z, marker="X", s=70, c=color,
                    depthshade=False, zorder=15)
        if i == leader_index_at_final:
            ax3.scatter(x, y, z, marker="o", s=240, facecolor="none",
                        edgecolor=LEADER_COLOR, linewidths=2.5,
                        depthshade=False)

    # ax3.scatter([], [], marker="o", facecolor="none", edgecolor=LEADER_COLOR,
    #             label="Leader")
    ax3.set_xlim(XLIM)
    ax3.set_ylim(YLIM)
    ax3.set_zlim(0, OBS_HEIGHT + 1)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    # box aspect: giữ tỉ lệ x:y thật, PHÓNG ĐẠI z cho khối dày hơn
    ax3.set_box_aspect((XLIM[1] - XLIM[0], YLIM[1] - YLIM[0],
                        (OBS_HEIGHT + 1) * Z_ASPECT))
    ax3.view_init(elev=ELEV_3D, azim=AZIM_3D)
    try:
        ax3.set_proj_type('persp')      # phối cảnh -> đỡ "dẹp"
    except Exception:
        pass
    # zoom vào cho bớt khoảng trắng (matplotlib >=3.3)
    try:
        ax3.dist = 8.0 / ZOOM_3D
    except Exception:
        pass
    if hasattr(ax3, "set_box_aspect"):
        try:
            ax3.margins(0)
        except Exception:
            pass
    if HIDE_3D_TICK_NUMBERS:
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_zticklabels([])
    # ax3.legend(loc="upper left", fontsize=8)
    # bóp lề trắng quanh axes
    fig3d.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    if export:
        plt.savefig("results/final_figure_3d.png", dpi=200)
        print("Saved results/final_figure_3d.png")

plt.show()