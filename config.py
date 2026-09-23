import os
import numpy as np

# ============================================================
# 1. COMMON PARAMETERS
# ------------------------------------------------------------
# Every value here can be overridden per scenario through
# SCENARIOS[N]['params'] (see section 3). Parameters that are
# derived from others (marked "derived") are computed in
# section 3 AFTER the overrides, unless the scenario sets them
# explicitly.
# ============================================================

# ─── Simulation ───
TIMESTEP = 0.1                 # control period (s)
HORIZON_LENGTH = 10            # MPC horizon (steps)
TARGET_ARRIVAL_TOL = 0.3       # stop when target is this close to its final point (m)

# ─── UAV physical ───
ROBOT_RADIUS = 0.2             # UAV radius (m)
VMAX = 2                       # max speed (m/s)
UMAX = 5                       # max acceleration (m/s^2)
SENSING_RADIUS = 3.0           # LiDAR range (m)
SENSING_NEIGHBOR = 3.0         # range within which other UAVs become hard constraints (m)
D_FRAC = 0.0                   # drag coefficient in the dynamics
VIEWING_RADIUS = 1.5           # half side of the square camera FOV (m); set per scenario
HFOV = 90
VFOV = 90

# ─── LiDAR ───
LIDAR_RANGE_MIN = 0.1          # (m)
LIDAR_ANGULAR_RES = np.pi / 90 # 2 deg per ray (rad)
LIDAR_NOISE = 0.01             # uniform range noise amplitude (m)
LIDAR_MARCH_STEP = 0.1         # ray-marching step for circular obstacles (m)

# ─── Occupancy grid + JPS planner ───
GRID_RESOLUTION = 0.5          # cell size (m)
INFLATE_MARGIN = 0.2           # INFLATE_RADIUS = ROBOT_RADIUS + INFLATE_MARGIN (derived)
GRID_RAY_BINS = 360            # angular bins used to integrate a scan into the grid
WORLD_BOUNDS = None            # (xmin, ymin, xmax, ymax); None -> from XLIM/YLIM (derived)
START_SNAP_RADIUS = 10.0       # search radius to move a blocked start to a free cell (m)
GOAL_SNAP_RADIUS = 30.0        # search radius to move a blocked goal to a free cell (m)
GOAL_CLAMP_MARGIN = 2.0        # goals outside the world are clamped this far inside (m)
PLANNER_TIME_BUDGET_MS = 30    # passed to the planner (JPS currently ignores it)

# ─── Path commit / replanning ───
MAX_COMMIT_AGE = 20            # force a replan after this many cycles
TARGET_REPLAN_THRESHOLD = 3.0  # replan if the (predicted) target moved more than this (m)
PATH_DEVIATION_THRESHOLD = 5.0 # replan if the UAV is farther than this from its path (m)
MAX_FAIL_BEFORE_STOP = 3       # consecutive planner failures before an emergency stop

# ─── Target estimation (Kalman) + lead pursuit ───
KF_PROCESS_NOISE_STD = 2.0     # unmodelled target acceleration (m/s^2)
KF_OBS_NOISE_STD = 0.3         # position measurement noise (m)
USE_LEAD_PURSUIT = True
LEAD_GAIN_MIN = 0.4
LEAD_GAIN_MAX = 1.0
VELOCITY_TRUSTED_THRESHOLD = 1.5   # use the velocity estimate only if its std < this (m/s)

# ─── RRT for the TARGET (legacy target mode) ───
TAR_STEP_LENGTH = 0.5
TAR_GOAL_SAMPLE_RATE = 0.1
TAR_MAX_ITER = 5000
TAR_RADIUS = 0.2               # target radius (m)
SAFETY_MARGIN = 0.4            # target-obstacle clearance (m)
TAR_EPSILON = 0.1

# ─── Target path smoothing ───
TAR_SMOOTH_ENABLE = True       # smooth the target path
TAR_SMOOTH_METHOD = "spline"   # "chaikin" | "spline" (RRT mode only)
TAR_SMOOTH_ITERATIONS = 4      # Chaikin iterations (method="chaikin")
TAR_SPLINE_DS = 0.4            # sample spacing along the spline (m)

# ─── MPC base weights (all modes) ───
W_tra = 1                      # track the planner's reference path
W_gui = 1                      # guidance terminal cost
W_u = 4e-1                     # control effort
W_corridor = 1.0               # barrier cost inside the safe corridor
CORRIDOR_BARRIER_EPS = 0.1     # barrier 1/(d + eps) regulariser (m)
CORRIDOR_BOX = None            # pydecomp local bounding box half size (m); None -> VIEWING_RADIUS (derived)
DT_CBF_GAMMA = 0.5             # discrete CBF decay rate (leader visibility)
CBF_BOX_RATIO = 0.1            # leader CBF keeps the target within +-ratio*VIEWING_RADIUS
STANDOFF_TIME = 1.0            # desired standoff = VIEWING_RADIUS - TAR_MAX_SPEED * STANDOFF_TIME (s)

# Weights that older scenarios only defined through 'params'
W_slack = 2.0
W_col = 1.0
W_form_dist = 1.0
W_sat_slot = 1.0

# ─── IPOPT ───
IPOPT_OPTIONS = {
    'ipopt.max_iter': 10000,
    'ipopt.print_level': 0,
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-2,
    'print_time': 0,
    'ipopt.acceptable_iter': 15,
}


# ============================================================
# 2-PHASE FORMATION PARAMETERS
# (SEARCH: no UAV sees the target / TRACK: leader + satellites)
# ============================================================

# ─── Mode constants ───
MODE_SEARCH = "SEARCH"
MODE_TRACK = "TRACK"

# ─── SEARCH mode ───
W_search_track = 1.0           # pull UAVs towards the predicted target

# ─── TRACK mode: satellites (target-centred formation) ───
W_sat_distance = 5.0
W_sat_angle = 5.0
W_sat_spread = 2.0
SAT_DISTANCE_RATIO = 1.9       # r_d = ratio * VIEWING_RADIUS
DESIRED_SEPARATION_RATIO = 0.5 # DESIRED_SEPARATION = ratio * VIEWING_RADIUS (derived)
SLOT_SPACING_RATIO = 5.6       # slot grid spacing = ratio * VIEWING_RADIUS

# ─── TRACK mode: leader ───
W_leader_slack = 1e3           # penalty on the CBF visibility slack

# ─── Dynamic anchor-based slots ───
ANCHOR_HYSTERESIS_RAD = 0.3    # ~17 deg: anchor changes only above this difference

# ─── Collision avoidance (soft, all modes) ───
W_collision_avoid = 5.0
COLLISION_AVOID_RATIO = 3.0    # COLLISION_AVOID_DISTANCE = ratio * ROBOT_RADIUS (derived)

# ─── Mode switch hysteresis (counter based) ───
K_IN_THRESHOLD = 3             # consecutive cycles seeing the target -> TRACK
K_OUT_THRESHOLD = 10           # consecutive cycles without the target -> SEARCH
K_HANDOFF_THRESHOLD = 5        # cycles before a leader handoff
VISIBILITY_MARGIN_RATIO = 0.1  # L_strict = L * (1 - ratio)
HANDOFF_DISTANCE_RATIO = 0.5   # delta_handoff = ratio * L

FORMATION_GAP_RATIO = 1.0          # FORMATION_GAP = ratio * VIEWING_RADIUS (derived)
TRACK_EXIT_HYSTERESIS_RATIO = 0.5  # TRACK_EXIT_HYSTERESIS = ratio * VIEWING_RADIUS (derived)
SLOT_INVALID_PATIENCE = 3
# Fixed slot assignment candidates
# GRID_CELLS = [(1, 0), (-1, 0), (0, 1), (0, -1),
#               (1, 1), (-1, 1), (-1, -1), (1, -1),
#               (2, 0), (-2, 0), (0, 2), (0, -2)]
# GRID_CELLS = [(0, 1), (0, -1)]
# GRID_CELLS = [(0, 1), (-1, 0)]
# GRID_CELLS = [(0, 1), (-1, 0), (0, -1)]
GRID_CELLS = [(0, 1), (-1, 0), (0, -1), (1, 0)]

SWITCH_MARGIN_RATIO = 0.5      # SWITCH_MARGIN = ratio * VIEWING_RADIUS (derived)
OPEN_ALL_SLOTS = False

# ============================================================
# 2. SCENARIOS 
# ============================================================

#     python main.py --scenario 3          (or SCENARIO=3 python main.py)
#     python plot_scenario.py 3 --traj
NUMBER_RUN=1
METHOD = 2  
SCENARIO = int(os.environ.get("SCENARIO", 9))

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
            # [10, 6, 3.],
            # [10, 2, 3.],
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

    9: dict(
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
            # [17, 3, 2, 2],
            # [24, 8.0, 2.5, 1.5],
            # [31.8, 5.2, 1.5, 2.5],
            # [13, 0.5, 2.5, 1.2],
            # [5, 0.5, 1, 1],
            # [40, 8, 2.5, 1.5],
            # [7, 8.5, 2, 1],
        ],
        circles=[
                #   [5.4, 5.8, 0.75],
                #   [24, 2.6, 1],
                #   [39, 6, 0.85],
                #   [42.0, 2, 0.75],
                #   [1.85, 8.6, 0.75],
                #   [33.4, 9.5, 0.85],
                #   [20, 1.88, 0.75]

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
        f"SCENARIO = {SCENARIO} does not exist. "
        f"Available scenarios: {sorted(SCENARIOS.keys())}")

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


# ─── Derived parameters ───
# Computed after the scenario is loaded so they follow the scenario's
# VIEWING_RADIUS / ROBOT_RADIUS / TAR_MAX_SPEED. A scenario can still pin any
# of them by putting the name in its 'params'.
def _derive(name, value):
    if name not in _overrides:
        globals()[name] = value


_derive("INFLATE_RADIUS", ROBOT_RADIUS + INFLATE_MARGIN)
_derive("COLLISION_AVOID_DISTANCE", COLLISION_AVOID_RATIO * ROBOT_RADIUS)
_derive("DESIRED_SEPARATION", DESIRED_SEPARATION_RATIO * VIEWING_RADIUS)
_derive("FORMATION_GAP", FORMATION_GAP_RATIO * VIEWING_RADIUS)
_derive("TRACK_EXIT_HYSTERESIS", TRACK_EXIT_HYSTERESIS_RATIO * VIEWING_RADIUS)
_derive("SWITCH_MARGIN", SWITCH_MARGIN_RATIO * VIEWING_RADIUS)
_derive("STANDOFF_DISTANCE", VIEWING_RADIUS - TAR_MAX_SPEED * STANDOFF_TIME)
if CORRIDOR_BOX is None:
    CORRIDOR_BOX = VIEWING_RADIUS
if WORLD_BOUNDS is None:
    WORLD_BOUNDS = (float(XLIM[0]), float(YLIM[0]), float(XLIM[1]), float(YLIM[1]))


# Output files
# RUN_FILE: point plot_*.py at a specific run, e.g.
#     RUN_FILE=runs/scen6_20260923-1530/data.pkl python plot_result.py
FILE_NAME = os.environ.get(
    "RUN_FILE", "data{}_scen{}_{}.txt".format(METHOD, SCENARIO, NUM_ROBOT))
FILE_NAME1 = FILE_NAME
SAVE_GIF = "results/data{}_scen{}_{}.gif".format(METHOD, SCENARIO, NUM_ROBOT)


# ============================================================
# 4. CONSISTENCY CHECKS
# ------------------------------------------------------------
# Many parameters are coupled (horizon reach vs corridor size,
# LiDAR ray spacing vs grid cell, ...). When scaling the map up,
# changing one value in isolation silently breaks another.
# validate_config() lists those problems; main.py logs them at
# start-up and refuses to run on ERROR.
#     python config.py            -> print the report for SCENARIO
# ============================================================
def _min_obstacle_gap(rects, circles):
    """Narrowest positive gap (m) between any two obstacles, or None."""
    gaps = []
    R = np.asarray(rects, float).reshape(-1, 4)
    C = np.asarray(circles, float).reshape(-1, 3)
    if len(R) > 1:
        x0, y0 = R[:, 0], R[:, 1]
        x1, y1 = x0 + R[:, 2], y0 + R[:, 3]
        dx = np.maximum(0, np.maximum(x0[:, None] - x1[None, :], x0[None, :] - x1[:, None]))
        dy = np.maximum(0, np.maximum(y0[:, None] - y1[None, :], y0[None, :] - y1[:, None]))
        d = np.hypot(dx, dy)[np.triu_indices(len(R), 1)]
        gaps.append(d)
    if len(C) > 1:
        d = np.linalg.norm(C[:, None, :2] - C[None, :, :2], axis=2) - C[:, None, 2] - C[None, :, 2]
        gaps.append(d[np.triu_indices(len(C), 1)])
    if len(R) and len(C):
        cx = np.clip(C[:, None, 0], R[None, :, 0], R[None, :, 0] + R[None, :, 2])
        cy = np.clip(C[:, None, 1], R[None, :, 1], R[None, :, 1] + R[None, :, 3])
        d = np.hypot(C[:, None, 0] - cx, C[:, None, 1] - cy) - C[:, None, 2]
        gaps.append(d.ravel())
    if not gaps:
        return None
    g = np.concatenate(gaps)
    g = g[g > 1e-9]                       # touching/overlapping obstacles act as one
    return float(g.min()) if g.size else None


def validate_config():
    """Return a list of (level, message); level is 'ERROR', 'WARNING' or 'INFO'."""
    out = []

    def add(level, msg):
        out.append((level, msg))

    reach = VMAX * HORIZON_LENGTH * TIMESTEP
    xmin, ymin, xmax, ymax = WORLD_BOUNDS

    # ── MPC horizon vs corridor / sensing ──
    if CORRIDOR_BOX < reach:
        add("WARNING", f"CORRIDOR_BOX={CORRIDOR_BOX:.2f} m < horizon reach "
                       f"VMAX*H*dt={reach:.2f} m: one corridor polytope is applied to the "
                       f"whole horizon, so the UAV is held back by the corridor wall.")
    if SENSING_RADIUS < reach:
        add("WARNING", f"SENSING_RADIUS={SENSING_RADIUS:.2f} m < horizon reach {reach:.2f} m: "
                       f"the MPC plans into space the LiDAR has not seen.")
    need = 2 * reach + 2 * ROBOT_RADIUS
    if SENSING_NEIGHBOR < need:
        add("WARNING", f"SENSING_NEIGHBOR={SENSING_NEIGHBOR:.2f} m < 2*reach + 2R = {need:.2f} m: "
                       f"two UAVs can close in before becoming each other's hard constraint.")
    if 2 * VMAX * TIMESTEP > 2 * ROBOT_RADIUS + 1e-9:
        add("WARNING", f"2*VMAX*dt={2 * VMAX * TIMESTEP:.2f} m > 2R={2 * ROBOT_RADIUS:.2f} m: "
                       f"UAVs can pass through each other between two MPC samples.")
    if VMAX / UMAX > HORIZON_LENGTH * TIMESTEP:
        add("WARNING", f"VMAX/UMAX={VMAX / UMAX:.2f} s is longer than the horizon "
                       f"{HORIZON_LENGTH * TIMESTEP:.2f} s: the UAV cannot reach top speed within it.")

    # ── Grid vs LiDAR ──
    ray_gap = SENSING_RADIUS * LIDAR_ANGULAR_RES
    if ray_gap > GRID_RESOLUTION:
        add("WARNING", f"LiDAR ray spacing at max range {ray_gap:.2f} m > GRID_RESOLUTION="
                       f"{GRID_RESOLUTION:.2f} m: obstacle surfaces get holes in the grid. "
                       f"Keep SENSING_RADIUS <= {GRID_RESOLUTION / LIDAR_ANGULAR_RES:.1f} m "
                       f"or refine LIDAR_ANGULAR_RES.")
    bin_gap = SENSING_RADIUS * 2 * np.pi / GRID_RAY_BINS
    if bin_gap > GRID_RESOLUTION:
        add("WARNING", f"GRID_RAY_BINS={GRID_RAY_BINS} gives {bin_gap:.2f} m between bins at max "
                       f"range > GRID_RESOLUTION: free space is under-carved.")
    if GRID_RESOLUTION > SENSING_RADIUS / 4:
        add("WARNING", f"GRID_RESOLUTION={GRID_RESOLUTION:.2f} m leaves fewer than 4 cells "
                       f"inside SENSING_RADIUS={SENSING_RADIUS:.2f} m.")
    if INFLATE_RADIUS < ROBOT_RADIUS:
        add("ERROR", f"INFLATE_RADIUS={INFLATE_RADIUS:.2f} m < ROBOT_RADIUS={ROBOT_RADIUS:.2f} m.")
    eff_inflate = np.ceil(INFLATE_RADIUS / GRID_RESOLUTION) * GRID_RESOLUTION
    if eff_inflate > 2 * INFLATE_RADIUS:
        add("INFO", f"Inflation is rounded up to whole cells: {INFLATE_RADIUS:.2f} m -> "
                    f"{eff_inflate:.2f} m effective.")
    cells = int(np.ceil((xmax - xmin) / GRID_RESOLUTION + 1) * np.ceil((ymax - ymin) / GRID_RESOLUTION + 1))
    add("INFO", f"Grid: {cells:,} cells per UAV "
                f"({cells * 5 * NUM_ROBOT / 1e6:.0f} MB for {NUM_ROBOT} UAVs).")

    # ── Target ──
    if TAR_MAX_SPEED >= VMAX:
        add("WARNING", f"TAR_MAX_SPEED={TAR_MAX_SPEED} >= VMAX={VMAX}: UAVs cannot catch up.")
    if STANDOFF_DISTANCE <= 0:
        add("WARNING", f"STANDOFF_DISTANCE={STANDOFF_DISTANCE:.2f} m <= 0 "
                       f"(VIEWING_RADIUS - TAR_MAX_SPEED*STANDOFF_TIME).")

    # ── World ──
    if (xmin, ymin, xmax, ymax) != (XLIM[0], YLIM[0], XLIM[1], YLIM[1]):
        add("WARNING", f"WORLD_BOUNDS={WORLD_BOUNDS} differs from XLIM/YLIM "
                       f"({XLIM}, {YLIM}).")

    def inside(p):
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax

    for i, s in enumerate(STARTS):
        if not inside(s):
            add("ERROR", f"UAV {i} start {s[:2].tolist()} is outside WORLD_BOUNDS.")
    for i, w in enumerate(TAR_WAYPOINTS):
        if not inside(w):
            add("ERROR", f"Target waypoint W{i} {list(np.asarray(w)[:2])} is outside WORLD_BOUNDS.")

    starts_xy = np.asarray(STARTS)[:, :2]
    for r in RECTANGLE_OBSTACLES:
        m = ((starts_xy[:, 0] >= r[0] - ROBOT_RADIUS) & (starts_xy[:, 0] <= r[0] + r[2] + ROBOT_RADIUS)
             & (starts_xy[:, 1] >= r[1] - ROBOT_RADIUS) & (starts_xy[:, 1] <= r[1] + r[3] + ROBOT_RADIUS))
        for i in np.flatnonzero(m):
            add("ERROR", f"UAV {i} starts inside rectangle obstacle {r.tolist()}.")
    for c in (OBSTACLES if len(OBSTACLES) else []):
        d = np.linalg.norm(starts_xy - c[:2], axis=1)
        for i in np.flatnonzero(d <= c[2] + ROBOT_RADIUS):
            add("ERROR", f"UAV {i} starts inside circle obstacle {c.tolist()}.")

    gap = _min_obstacle_gap(_s['rects'], _circles)
    if gap is not None:
        passable = 2 * eff_inflate + GRID_RESOLUTION
        level = "INFO" if gap >= passable else "WARNING"
        add(level, f"Narrowest gap between obstacles: {gap:.2f} m "
                   f"(the grid planner needs about > {passable:.2f} m to pass through a gap).")
    return out


if __name__ == "__main__":
    print(f"Scenario {SCENARIO}: {NUM_ROBOT} UAV, world {WORLD_BOUNDS}, "
          f"horizon reach {VMAX * HORIZON_LENGTH * TIMESTEP:.2f} m")
    for level, msg in validate_config():
        print(f"[{level}] {msg}")