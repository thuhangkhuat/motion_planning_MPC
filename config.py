import os
import numpy as np

# ============================================================
# 1. COMMON PARAMETERS
# ============================================================

# ─── Simulation ───
TIMESTEP = 0.1                 # chu kỳ điều khiển (s)
HORIZON_LENGTH = 10            # số bước MPC horizon

# ─── UAV physical ───
ROBOT_RADIUS = 0.2             # bán kính UAV (m)
VMAX = 2                       # tốc độ tối đa (m/s)
UMAX = 5                      # gia tốc tối đa (m/s²)
SENSING_RADIUS = 3.0          # tầm cảm biến lidar (m)
SENSING_NEIGHBOR = 3.0         # tầm "thấy" UAV khác (m)
D_FRAC = 0.0                   # hệ số drag trong dynamics
VIEWING_RADIUS = 1.5
HFOV = 90
VFOV = 90

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
W_sat_distance = 5.0           # giữ khoảng cách r_d tới TARGET
W_sat_angle = 5.0              # slot angle quanh TARGET (cos-based)
W_sat_spread = 2.0             # break symmetry, đẩy satellites tản ra
SAT_DISTANCE_RATIO = 1.9       # r_d = ratio * VIEWING_RADIUS
DESIRED_SEPARATION = 0.5 * VIEWING_RADIUS   # d_form = ratio * VIEWING_RADIUS
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

FORMATION_GAP = VIEWING_RADIUS
TRACK_EXIT_HYSTERESIS = 0.5 * VIEWING_RADIUS 
SLOT_INVALID_PATIENCE = 3
#fix slot assignment
# GRID_CELLS = [(1, 0), (-1, 0), (0, 1), (0, -1),
#               (1, 1), (-1, 1), (-1, -1), (1, -1),
#               (2, 0), (-2, 0), (0, 2), (0, -2)]
# GRID_CELLS = [(0, 1), (0, -1)]
# GRID_CELLS = [(0, 1), (-1, 0)]
# GRID_CELLS = [(0, 1), (-1, 0), (0, -1)]
GRID_CELLS = [(0, 1), (-1, 0), (0, -1), (1, 0)]

SWITCH_MARGIN   = 0.5 * VIEWING_RADIUS   # biên hysteresis: chỉ đổi slot khi rẻ hơn ngần này
OPEN_ALL_SLOTS  = False 
# ============================================================
# 2. SCENARIOS 
# ============================================================

#     SCENARIO=3 python main.py
#     SCENARIO=3 python plot_scenario.py --traj
NUMBER_RUN=1
METHOD = 2  
SCENARIO = int(os.environ.get("SCENARIO", 2))

SCENARIOS = {

    # ────────────────────────────────────────────────────────
    # Scenario 1:
    # ────────────────────────────────────────────────────────
    1: dict(
        tar_max_speed=1,
        viewing_radius=1.5,
        waypoints=[
            np.array([4.0, 2.2, 0]),
            np.array([18, 7.0, 0]),
            np.array([33, 3, 0]),
            np.array([47, 5.5, 0]),
        ],
        starts=np.array([
            [3, 2, 3.],
            [10, 6, 3.],
            [10, 2, 3.],
        ]),
        rects=
        [
            [7, 7, 2.5, 1.5],
            [20, 2, 1.5, 2.5],
            [26, 0.5, 2, 2],
            [37, 6, 2, 2.5],
        ],
        circles=[
                 [14, 8.5, 1],
                #  [46, 1.5, 1],
                 [30, 6, 1],
                #  [43, 7, 1],
                 [15, 2, 1],
                 [2.5, 5.5, 1],
                #  [36, 2.3, 0.75],
                #  [23, 8, 0.75],
                 ],
        xlim=[0, 50],
        ylim=[0, 10],

            params=dict(                    # (optional)
                W_slack=2.0,
                W_col=1.0,
                W_form_dist=1.0,
                W_sat_slot = 1.0,
            ),
        
    ),

    2: dict(
        tar_max_speed=1,
        viewing_radius=1.5,
        waypoints=[
            np.array([4.0, 2.2, 0]),
            np.array([18, 7.0, 0]),
            np.array([33, 3, 0]),
            np.array([47, 5.5, 0]),
        ],
        starts=np.array([
            [3, 2, 3.],
            [10, 6, 3.],
            [10, 2, 3.],
        ]),
        rects=
        [
            [7, 7, 2.5, 1.5],
            [20, 2, 1.5, 2.5],
            [26, 0.5, 2, 2],
            [37, 6, 2, 2.5],
            [2.3, 7.5, 2, 2],
            [30, 8, 2.5, 1.2],
            [16, 0.5, 3, 1.2],
            [1, 2.0, 1.2, 3],
            [23.4, 6.6,0.8,0.8],
            [8.8, 2.8, 0.8, 0.8]
        ],
        circles=[
                 [14, 8.5, 1],
                 [30, 6, 1],
                 [16, 3.7, 0.75],
                 [4.5, 5.5, 1],
                 [20, 8, 0.6],
                 [33.5, 1.9, 0.7],
                 [13.5, 1.6, 0.8],
                 [26, 8.75, 0.8],
                 [23.3, 2.2, 0.8],
                 [33.6, 6.8, 0.75],

                 ],
        xlim=[-2, 51],
        ylim=[-1, 11],

            params=dict(                    # (optional)
                W_slack=2.0,
                W_col=1.0,
                W_form_dist=1.0,
                W_sat_slot = 1.0,
            ),
        
    ),

    3: dict(
        tar_max_speed=1,
        viewing_radius=1.5,
        waypoints=[
            np.array([4.0, 2.2, 0]),
            np.array([18, 7.0, 0]),
            np.array([33, 3, 0]),
            np.array([47, 5.5, 0]),
        ],
        starts=np.array([
            [3, 2, 3.],
            [14, 1, 3.],
            [2, 8, 3.],
        ]),
        rects=[
             [3.5, 4.2, 1.5, 1.5],
             [15.2, 3.3, 2.5, 1.5],
             [20, 2, 2, 2],
             [40, 5.8, 1.5, 2.5],
        ],
        circles=[
                 [10, 2, 1],
                 [11.3, 8.0, 0.75],
                 [25.4, 8.75, 1],
                 [32, 6, 0.85],
                 [27, 2, 1],
                 [45, 2.5, 1],
                 ],
        xlim=[0, 50],
        ylim=[0, 10],

            params=dict(                    # (optional)
                W_slack=2.0,
                W_col=1.0,
                W_form_dist=1.0,
                W_sat_slot = 1.2,
            ),
        
    ),

    4: dict(
        tar_max_speed=1,
        viewing_radius=1.5,
        waypoints=[
            np.array([4.0, 2.2, 0]),
            np.array([18, 7.0, 0]),
            np.array([33, 3, 0]),
            np.array([47, 5.5, 0]),
        ],
        starts=np.array([
            [3, 2, 3.],
            [14, 1, 3.],
            [2, 8, 3.],
        ]),
        rects=[
             [3.5, 4.2, 1.5, 1.5],
             [15.2, 4.0, 2.5, 1.5],
             [20, 2, 2, 2],
             [40, 5.8, 1.5, 2.5],
             [35, 7.5, 2, 2],
             [29, 8.0, 2.5, 1],
             [35, 0.9, 2.5, 1.5],
        ],
        circles=[
                 [10, 2, 1],
                 [11.3, 8.0, 0.75],
                 [25.4, 8.75, 1],
                 [32, 6, 0.85],
                 [27, 2, 1],
                 [45, 2.5, 1],
                 [18, 8.2, 0.75],
                 [15.9, 2, 0.85]

                 ],
        xlim=[-1, 51],
        ylim=[-1, 11],

            params=dict(                    # (optional)
                W_slack=2.0,
                W_col=1.0,
                W_form_dist=1.0,
                W_sat_slot = 1.2,
            ),
        
    ),

    5: dict(
        tar_max_speed=1,
        viewing_radius=1.5,
        waypoints=[
            np.array([4.0, 2.2, 0]),
            np.array([18, 7.0, 0]),
            np.array([33, 3, 0]),
            np.array([47, 5.5, 0]),
        ],
        starts=np.array([
            [3, 2, 3.],
            [10, 8, 3.],
            [11, 2, 3.],
            [1.5, 6, 3.],
        ]),
        rects=[
            [5, 0.5, 1.5, 1.5],
            [16, 8.5, 2.5, 1],
            [24, 1.0, 2, 2],
            [30, 6, 1.5, 2.5],
            # [2, 7.5, 2, 2],
            # [32, 7, 2, 2],
        ],
        circles=[
                  [6.5, 6.0, 0.75],
                  [18, 4, 1],
                  [39, 6, 1],
                #  [4.5, 5.5, 1],
                #  [20, 7.4, 0.75],
                #  [46, 1.5, 1],
                #  [43, 7, 1],
                #  [35, 2.5, 0.75],
                 ],
        xlim=[0, 50],
        ylim=[0, 10],

            params=dict(                    # (optional)
                W_slack=2.0,
                W_col=1.0,
                W_form_dist=1.0,
                W_sat_slot = 1.0,
            ),
        
    ),

    6: dict(
        tar_max_speed=1,
        viewing_radius=1.5,
        waypoints=[
            np.array([4.0, 2.2, 0]),
            np.array([18, 7.0, 0]),
            np.array([33, 3, 0]),
            np.array([47, 5.5, 0]),
        ],
        starts=np.array([
            [3, 2, 3.],
            [10, 8, 3.],
            [11, 2, 3.],
            [1.5, 6, 3.],
        ]),
        rects=[
            [5, 0.5, 1.5, 1.5],
            [16, 8.5, 2.5, 1],
            [24, 1.0, 2, 2],
            [30, 6, 1.5, 2.5],
            [2.5, 7.7, 2.5, 1.5],
            [14.2, 1.0, 2.5, 1.5],
            [34, 7.2, 2, 2],
            [3, 4.3, 1.5, 1.5]
        ],
        circles=[
                  [6.5, 6.0, 0.75],
                  [18, 4, 1],
                  [39, 6, 1],
                  [27.0, 8.5, 0.85],
                  [33, 1.8, 0.65],
                  [21.3, 2.7, 0.75],
                  [34, 5.8, 0.65],
                  [23.9, 6.9, 0.55],
                 ],
        xlim=[-1, 51],
        ylim=[-1, 11],

            params=dict(                    # (optional)
                W_slack=2.0,
                W_col=1.0,
                W_form_dist=1.0,
                W_sat_slot = 1.0,
            ),
        
    ),

    7: dict(
        tar_max_speed=1,
        viewing_radius=1.5,
        waypoints=[
            np.array([4.0, 2.2, 0]),
            np.array([18, 7.0, 0]),
            np.array([33, 3, 0]),
            np.array([47, 5.5, 0]),
        ],
        starts=np.array([
            [3, 2, 3.],
            [10, 8, 3.],
            [11, 2, 3.],
            [1.5, 6, 3.],
            [5, 8.5, 3.]
        ]),
        rects=[
            [16.3, 2.5, 1.5, 1.5],
            [24, 8.0, 2.5, 1.5],
            [31.8, 5.2, 1.5, 2.5],
        ],
        circles=[
                  [5.4, 5.8, 0.75],
                  [24, 2.4, 1],
                  [39, 6, 0.85],
                  [42.0, 2, 0.55],
                 ],
        xlim=[0, 50],
        ylim=[0, 10],

            params=dict(                    # (optional)
                W_slack=2.0,
                W_col=1.0,
                W_form_dist=1.0,
                W_sat_slot = 1.0,
            ),
        
    ),

    8: dict(
        tar_max_speed=1,
        viewing_radius=1.5,
        waypoints=[
            np.array([4.0, 2.2, 0]),
            np.array([18, 7.0, 0]),
            np.array([33, 3, 0]),
            np.array([47, 5.5, 0]),
        ],
        starts=np.array([
            [3, 2, 3.],
            [10, 8, 3.],
            [11, 2, 3.],
            [1.5, 6, 3.],
            [5, 8.5, 3.]
        ]),
        rects=[
            [17, 3, 2, 2],
            [24, 8.0, 2.5, 1.5],
            [31.8, 5.2, 1.5, 2.5],
            [13, 0.5, 2.5, 1.2],
            [5, 0.5, 1, 1],
            [40, 8, 2.5, 1.5],
            [7, 8.5, 2, 1],
        ],
        circles=[
                  [5.4, 5.8, 0.75],
                  [24, 2.6, 1],
                  [39, 6, 0.85],
                  [42.0, 2, 0.75],
                  [1.85, 8.6, 0.75],
                  [33.4, 9.5, 0.85],
                  [20, 1.88, 0.75]

                 ],
        xlim=[-1, 51],
        ylim=[-1, 11],

            params=dict(                    # (optional)
                W_slack=2.0,
                W_col=1.0,
                W_form_dist=1.0,
                W_sat_slot = 1.0,
            ),
        
    ),
   
   

    # ────────────────────────────────────────────────────────
    # TEMPLATE
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
    #     rects=[[x, y, w, h], ...],    
    #     circles=[[cx, cy, r], ...],    
    #     xlim=[0, 50],
    #     ylim=[0, 50],
    #     params=dict(                   
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
# FILE_NAME = "data{}_run_{}_scen{}_{}.txt".format(METHOD, NUMBER_RUN, SCENARIO, NUM_ROBOT)
FILE_NAME = "data{}_scen{}_{}.txt".format(METHOD, SCENARIO, NUM_ROBOT)
FILE_NAME1 = FILE_NAME
SAVE_GIF = "results/data{}_scen{}_{}.gif".format(METHOD, SCENARIO, NUM_ROBOT)