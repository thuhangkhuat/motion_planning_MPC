import os
import numpy as np

# ============================================================
# 1. COMMON PARAMETERS
# ============================================================

# ─── Simulation ───
TIMESTEP = 0.1                 # chu kỳ điều khiển (s)
HORIZON_LENGTH = 10            # số bước MPC horizon

# ─── UAV physical ───
ROBOT_RADIUS = 0.3             # bán kính UAV (m)
VMAX = 5                       # tốc độ tối đa (m/s)
UMAX = 20                      # gia tốc tối đa (m/s²)
SENSING_RADIUS = 10.0          # tầm cảm biến lidar (m)
SENSING_NEIGHBOR = 5.0         # tầm "thấy" UAV khác (m)
D_FRAC = 0.0                   # hệ số drag trong dynamics

# ─── Planner cho UAV ───
METHOD = 2                     # 1: A*, 2: JPS, 3: RRT
# RRT-Connect (chỉ dùng khi METHOD == 3)
STEP_LENGTH = 0.1
GOAL_SAMPLE_RATE = 0.01
MAX_ITER = 5000

# ─── RRT cho TARGET (sinh trajectory mục tiêu) ───
TAR_STEP_LENGTH = 0.5
TAR_GOAL_SAMPLE_RATE = 0.1
TAR_MAX_ITER = 5000
TAR_RADIUS = 0.2               # bán kính target
SAFETY_MARGIN = 0.4            # khoảng đệm target - obstacle
TAR_EPSILON = 0.1

# ─── Target path smoothing ───
TAR_SMOOTH_ENABLE = True       # bật/tắt làm mượt path target
TAR_SMOOTH_METHOD = "spline"   # "chaikin": vạt góc | "spline": Catmull-Rom (mượt hơn)
TAR_SMOOTH_ITERATIONS = 4      # số lần lặp Chaikin (chỉ dùng khi method="chaikin")
TAR_SPLINE_DS = 0.4            # khoảng cách sample trên spline (m), nhỏ = mịn hơn

# ─── MPC base weights (mọi mode) ───
W_tra = 1                      # tracking reference path (planner output)
W_gui = 1                      # guidance terminal cost
W_u = 4e-1                     # control effort
W_corridor = 1.0               # barrier cost trong safe corridor
DT_CBF_GAMMA = 0.5             # CBF discrete gamma (visibility leader)

# ============================================================
# 2-PHASE FORMATION PARAMETERS
# (SEARCH: chưa UAV nào thấy target / TRACK: có leader + satellites)
# ============================================================

# ─── Mode constants ───
MODE_SEARCH = "SEARCH"
MODE_TRACK = "TRACK"

# ─── SEARCH mode ───
W_search_track = 1.0           # kéo UAV về predicted target

# ─── TRACK mode: satellites (target-centered formation) ───
W_sat_distance = 0.5           # giữ khoảng cách r_d tới TARGET
W_sat_angle = 2.0              # slot angle quanh TARGET (cos-based)
W_sat_spread = 1.0             # break symmetry, đẩy satellites tản ra
SAT_DISTANCE_RATIO = 0.5       # r_d = ratio * VIEWING_RADIUS

# ─── TRACK mode: leader ───
W_leader_slack = 1e3           # phạt slack CBF visibility

# ─── Slot dynamic anchor-based ───
ANCHOR_HYSTERESIS_RAD = 0.3    # ~17°: anchor chỉ đổi khi chênh > ngưỡng

# ─── Collision avoidance (soft, mọi mode) ───
W_collision_avoid = 5.0        # weight safety preference
COLLISION_AVOID_DISTANCE = 3 * ROBOT_RADIUS   # d_safe = 0.9m (1.5x hard limit 2R)

# ─── Mode switch hysteresis (counter-based) ───
K_IN_THRESHOLD = 3             # cycles liên tiếp thấy target → vào TRACK
K_OUT_THRESHOLD = 10           # cycles liên tiếp mất target → về SEARCH
K_HANDOFF_THRESHOLD = 5        # cycles để handoff leader
VISIBILITY_MARGIN_RATIO = 0.1  # L_strict = L * (1 - ratio)
HANDOFF_DISTANCE_RATIO = 0.5   # Δ_handoff = ratio * L


# ============================================================
# 2. SCENARIOS 
# ============================================================

#     SCENARIO=3 python main.py
#     SCENARIO=3 python plot_scenario.py --traj
SCENARIO = int(os.environ.get("SCENARIO", 1))

SCENARIOS = {

    # ────────────────────────────────────────────────────────
    # Scenario 1:
    # ────────────────────────────────────────────────────────
    1: dict(
        tar_max_speed=8,
        viewing_radius=30,
        waypoints=[
            np.array([16.0, 20.0, 0]),
            np.array([300.0, 300.0, 0]),
            np.array([145.0, 465.0, 0]),
            np.array([450.0, 150.0, 0]),
        ],
        starts=np.array([
            [20, 19, 3.],
            [40, 50, 3.],
            [0, 0, 3.],
        ]),
        rects=[
            [70, 120, 40, 70],
            [200, 230, 50, 80],
            [300, 100, 80, 60],
            [170, 80, 70, 50],
            [345, 235, 90, 70],
            [380, 350, 60, 80],
            [170, 340, 70, 50],
            [70, 370, 50, 70],
            [220, 430, 50, 50],
            [50, 240, 60, 60],
            [390, 35, 60, 60],
        ],
        circles=[],
        xlim=[0, 500],
        ylim=[0, 500],
    ),

    # ────────────────────────────────────────────────────────
    # Scenario 2:
    # ────────────────────────────────────────────────────────
    2: dict(
        tar_max_speed=2,
        viewing_radius=5,
        waypoints=[
            np.array([3, 10, 0]),
            np.array([30.0, 30.0, 0]),
            np.array([42, 46, 0]),
            np.array([45.0, 15.0, 0]),
        ],
        starts=np.array([
            [15.0, 30.0, 3.],
            [4.0, 5.0, 3.],
            [10.0, 5.0, 3.],
            [20.0, 16.0, 3.],
            [3.0, 20.0, 3.],
        ]),
        rects=[
            [7.0, 12.0, 3.0, 5.0],
            [20.0, 23.0, 5.0, 8.0],
            [30.0, 10.0, 8.0, 6.0],
            [17.0, 8.0, 7.0, 5.0],
            [34.5, 23.5, 9.0, 7.0],
            [38.0, 35.0, 6.0, 8.0],
            [17.0, 34.0, 7.0, 5.0],
            [7.0, 37.0, 5.0, 7.0],
            [22.0, 43.0, 5.0, 5.0],
            [5.0, 24.0, 6.0, 6.0],
            [39.0, 3.5, 6.0, 6.0],
        ],
        circles=[],
        xlim=[0, 50],
        ylim=[0, 50],
    ),

    # ────────────────────────────────────────────────────────
    # Scenario 3:
    # ────────────────────────────────────────────────────────
    3: dict(
        tar_max_speed=2,
        viewing_radius=5,
        waypoints=[
            np.array([3, 10, 0]),
            np.array([30.0, 30.0, 0]),
            np.array([42, 46, 0]),
            np.array([45.0, 15.0, 0]),
        ],
        starts=np.array([
            [20.0, 20.0, 3.],
            [4.0, 5.0, 3.],
            [10.0, 5.0, 3.],
        ]),
        rects=[
            [7.0, 12.0, 3.0, 5.0],
            [20.0, 23.0, 5.0, 8.0],
            [30.0, 10.0, 8.0, 6.0],
            [17.0, 8.0, 7.0, 5.0],
            [34.5, 23.5, 9.0, 7.0],
            [38.0, 35.0, 6.0, 8.0],
            [17.0, 34.0, 7.0, 5.0],
            [7.0, 37.0, 5.0, 7.0],
            [22.0, 43.0, 5.0, 5.0],
            [5.0, 24.0, 6.0, 6.0],
            [39.0, 3.5, 6.0, 6.0],
        ],
        circles=[],
        xlim=[0, 50],
        ylim=[0, 50],
    ),

    # ────────────────────────────────────────────────────────
    # Scenario 4: TEMPLATE — copy đoạn này để tạo kịch bản mới
    # ────────────────────────────────────────────────────────
    # 4: dict(
    #     tar_max_speed=2,
    #     viewing_radius=5,
    #     waypoints=[
    #         np.array([x, y, 0]),
    #         ...
    #     ],
    #     starts=np.array([
    #         [x, y, 3.],
    #         ...
    #     ]),
    #     rects=[[x, y, w, h], ...],      # chữ nhật / vuông
    #     circles=[[cx, cy, r], ...],     # tròn
    #     xlim=[0, 50],
    #     ylim=[0, 50],
    #     params=dict(                    # (optional) tune riêng scenario này
    #         W_sat_angle=5.0,
    #         W_collision_avoid=2.0,
    #         K_OUT_THRESHOLD=15,
    #     ),
    # ),
}


# ============================================================
# 3. EXPORT
# ============================================================

if SCENARIO not in SCENARIOS:
    raise ValueError(
        f"SCENARIO = {SCENARIO} không tồn tại. "
        f"Các scenario có sẵn: {sorted(SCENARIOS.keys())}")

_s = SCENARIOS[SCENARIO]

# ─── Per-scenario parameter overrides ───
_overrides = _s.get('params', {})
globals().update(_overrides)

# Target
TAR_MAX_SPEED = _s['tar_max_speed']
TAR_WAYPOINTS = _s['waypoints']

# Environment
VIEWING_RADIUS = _s['viewing_radius']
XLIM = _s['xlim']
YLIM = _s['ylim']

# UAV
STARTS = _s['starts']
NUM_ROBOT = STARTS.shape[0]
GOALS = STARTS + np.array([21., 0., 0.])

RECTANGLE_OBSTACLES = [np.array(r, dtype=float) for r in _s['rects']]

POLYGON_OBSTACLES = [
    np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    for (x, y, w, h) in _s['rects']
]

_circles = _s.get('circles', [])
OBSTACLES = np.array(_circles, dtype=float) if _circles else np.array([])

# Output files
FILE_NAME = "data{}_scen{}_{}.txt".format(METHOD, SCENARIO, NUM_ROBOT)
FILE_NAME1 = FILE_NAME
SAVE_GIF = "results/data{}_scen{}_{}.gif".format(METHOD, SCENARIO, NUM_ROBOT)