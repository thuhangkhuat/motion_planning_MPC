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
LIDAR_MARCH_STEP = 0.1         # ray-marching step for circular obstacles (m, mode "march")
LIDAR_MODE = "analytic"        # "analytic" (exact, vectorised) | "march" (original)

# ─── Occupancy grid + JPS planner ───
GRID_RESOLUTION = 0.5          # cell size (m)
INFLATE_MARGIN = 0.2           # INFLATE_RADIUS = ROBOT_RADIUS + INFLATE_MARGIN (derived)
GRID_RAY_BINS = 360            # angular bins used to integrate a scan into the grid
WORLD_BOUNDS = None            # (xmin, ymin, xmax, ymax); None -> from XLIM/YLIM (derived)
START_SNAP_RADIUS = 10.0       # search radius to move a blocked start to a free cell (m)
GOAL_SNAP_RADIUS = 30.0        # search radius to move a blocked goal to a free cell (m)
GOAL_CLAMP_MARGIN = 2.0        # goals outside the world are clamped this far inside (m)
PLANNER_TIME_BUDGET_MS = 30    # passed to the planner (JPS currently ignores it)
PLANNER = "local"              # "local" (planner_grid.py) | "jps" (original, whole grid)
LOCAL_PLAN_RADIUS = 100.0      # half size of the planning window around the UAV (m)
UNKNOWN_POLICY = "blocked"     # "blocked": unseen cells cannot be entered
                               # "optimistic": unseen cells cost UNKNOWN_COST x a free step
UNKNOWN_COST = 1.0

# ─── Path commit / replanning ───
MAX_COMMIT_AGE = 20            # force a replan after this many cycles
TARGET_REPLAN_THRESHOLD = 3.0  # replan if the (predicted) target moved more than this (m)
PATH_DEVIATION_THRESHOLD = 5.0 # replan if the UAV is farther than this from its path (m)
MAX_FAIL_BEFORE_STOP = 3       # consecutive planner failures before an emergency stop
FAILSAFE_BRAKE = True          # emergency stop / repeated MPC failure -> brake at UMAX
                               # (False = original: zero acceleration, i.e. keep flying)
MPC_FAILS_BEFORE_BRAKE = 1     # consecutive MPC failures that still reuse the old plan

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

# ─── MPC implementation / cost scaling ───
MPC_BACKEND = "parametric"     # "parametric" (built once, mpc_problem.py) | "rebuild" (original)
CORRIDOR_MAX_FACES = 16        # corridor faces the parametric MPC is built for (grows if needed)
COST_LENGTH_SCALE = 1.0        # distances enter the cost divided by this (m); 1 = original cost
COST_ACCEL_SCALE = 1.0         # controls enter the cost divided by this (m/s^2); 1 = original

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
# ------------------------------------------------------------
# Scenarios live in scenarios/*.yaml (see scenarios/README.md).
#     python main.py --scenario 6          -> scenarios/scen6.yaml
#     python main.py --scenario big1000    -> scenarios/big1000.yaml
# Points picked with pick_waypoints.py (target waypoints / speeds,
# UAV starts) are stored next to it in <name>.picks.json and
# override the YAML values.
# ============================================================
NUMBER_RUN = 1
METHOD = 2
if __name__ == "__main__" and len(__import__("sys").argv) > 1:    # python config.py <scenario>
    os.environ["SCENARIO"] = __import__("sys").argv[1]
SCENARIO = os.environ.get("SCENARIO", "9")
SCENARIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")
START_ALTITUDE = 3.0           # z used when a start is given as [x, y]


def scenario_path(name):
    """scenarios/scen<name>.yaml for numbers, scenarios/<name>.yaml otherwise."""
    name = str(name)
    cands = [f"scen{name}.yaml", f"{name}.yaml", name]
    for c in cands:
        p = c if os.path.isabs(c) else os.path.join(SCENARIO_DIR, c)
        if os.path.isfile(p):
            return p
    raise ValueError(f"Scenario '{name}' not found in {SCENARIO_DIR}. "
                     f"Available: {list_scenarios()}")


def list_scenarios():
    if not os.path.isdir(SCENARIO_DIR):
        return []
    names = [f[:-5] for f in sorted(os.listdir(SCENARIO_DIR)) if f.endswith(".yaml")]
    return [n[4:] if n.startswith("scen") and n[4:].isdigit() else n for n in names]


def picks_path(name):
    return scenario_path(name)[:-5] + ".picks.json"


def load_scenario(name):
    """Raw scenario dict (YAML) with <name>.picks.json merged in."""
    import json
    import yaml
    path = scenario_path(name)
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    d.setdefault("target", {})
    d.setdefault("obstacles", {})
    d["_file"] = path
    d["_picks"] = None
    picks = path[:-5] + ".picks.json"
    legacy = os.path.join(SCENARIO_DIR, f"target_scen{name}.json")   # older picker format
    if os.path.isfile(picks):
        with open(picks, encoding="utf-8") as f:
            p = json.load(f)
        if p.get("target", {}).get("waypoints"):
            d["target"]["waypoints"] = p["target"]["waypoints"]
            d["target"]["speeds"] = p["target"].get("speeds")
        if p.get("starts"):
            d["starts"] = p["starts"]
        d["_picks"] = picks
    elif os.path.isfile(legacy):
        with open(legacy, encoding="utf-8") as f:
            p = json.load(f)
        d["target"]["waypoints"] = p["waypoints"]
        d["target"]["speeds"] = p.get("speeds")
        d["_picks"] = legacy
    return d


# ============================================================
# 3. EXPORT
# ============================================================
_s = load_scenario(SCENARIO)
SCENARIO_NAME = os.path.basename(_s["_file"])[:-5]      # e.g. "scen6", "big1000"

# ─── Per-scenario parameter overrides ───
_overrides = _s.get('params') or {}
globals().update(_overrides)

# Environment
VIEWING_RADIUS = float(_s['viewing_radius'])
XLIM = [float(v) for v in _s['xlim']]
YLIM = [float(v) for v in _s['ylim']]

# Target
TAR_MAX_SPEED = _s['tar_max_speed']
TAR_WAYPOINTS = [np.array(list(w) + [0.0] * (3 - len(w)), dtype=float)
                 for w in _s['target'].get('waypoints', [])]
TAR_SPEEDS = (np.asarray(_s['target']['speeds'], float)
              if _s['target'].get('speeds') else None)

# UAV
STARTS = np.array([list(p) + [START_ALTITUDE] * (3 - len(p)) for p in _s['starts']],
                  dtype=float)
NUM_ROBOT = STARTS.shape[0]
# Robot.goal is overwritten with the target position every step; the initial
# value only matters before the first update.
GOALS = STARTS.copy()

# Obstacles: explicit ones from the file + generated ones (map_gen.py)
_rects = [list(map(float, r)) for r in (_s['obstacles'].get('rects') or [])]
_circles = [list(map(float, c)) for c in (_s['obstacles'].get('circles') or [])]
GENERATION_STATS = None
if _s.get('generate'):
    from map_gen import generate_obstacles
    from geometry import catmull_rom
    _keep = [p[:2] for p in STARTS] + [w[:2] for w in TAR_WAYPOINTS]
    # the path that path_clearance protects = the path the target will follow
    _tpath = np.array([w[:2] for w in TAR_WAYPOINTS]).reshape(-1, 2)
    if TAR_SMOOTH_ENABLE and len(_tpath) >= 3:
        _tpath = catmull_rom(_tpath, TAR_SPLINE_DS)[0]
    _rects, _circles, GENERATION_STATS = generate_obstacles(
        _s['generate'], XLIM, YLIM, keep_clear_points=_keep,
        fixed_rects=_rects, fixed_circles=_circles,
        target_path=_tpath)

RECTANGLE_OBSTACLES = [np.array(r, dtype=float) for r in _rects]
POLYGON_OBSTACLES = [
    np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    for (x, y, w, h) in _rects
]
OBSTACLES = np.array(_circles, dtype=float) if _circles else np.array([])

# Resolved scenario (what the simulation actually uses) -> run snapshot, plots
SCENARIO_DEF = {
    "name": SCENARIO_NAME,
    "file": _s["_file"],
    "picks_file": _s["_picks"],
    "description": _s.get("description", ""),
    "xlim": XLIM, "ylim": YLIM,
    "viewing_radius": VIEWING_RADIUS,
    "tar_max_speed": TAR_MAX_SPEED,
    "waypoints": [w.tolist() for w in TAR_WAYPOINTS],
    "speeds": None if TAR_SPEEDS is None else TAR_SPEEDS.tolist(),
    "starts": STARTS.tolist(),
    "rects": _rects, "circles": _circles,
    "generate": _s.get("generate"),
    "params": dict(_overrides),
}


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

    # ── Planner / MPC implementation ──
    if PLANNER == "local":
        if LOCAL_PLAN_RADIUS < SENSING_RADIUS:
            add("WARNING", f"LOCAL_PLAN_RADIUS={LOCAL_PLAN_RADIUS} m < SENSING_RADIUS="
                           f"{SENSING_RADIUS} m: the planner ignores part of what the LiDAR sees.")
        span = max(xmax - xmin, ymax - ymin)
        if UNKNOWN_POLICY == "blocked" and span > 2 * LOCAL_PLAN_RADIUS:
            add("WARNING", "UNKNOWN_POLICY='blocked' on a map larger than the planning "
                           "window: a target far outside the explored area can only be "
                           "approached, not planned to. Use 'optimistic' on large maps.")
        side = 2 * int(np.ceil(LOCAL_PLAN_RADIUS / GRID_RESOLUTION)) + 1
        nx = int(np.ceil((xmax - xmin) / GRID_RESOLUTION)) + 1
        ny = int(np.ceil((ymax - ymin) / GRID_RESOLUTION)) + 1
        add("INFO", f"Local planning window: {min(side, nx) * min(side, ny):,} cells"
                    + (" (covers the whole map)" if side >= max(nx, ny) else "") + ".")
        try:
            import numba  # noqa: F401
        except ImportError:
            add("WARNING", "numba is not installed: the local planner runs in pure Python "
                           "(10-100x slower). pip install numba")
    elif PLANNER == "jps":
        cells_ = np.ceil((xmax - xmin) / GRID_RESOLUTION + 1) * np.ceil((ymax - ymin) / GRID_RESOLUTION + 1)
        if cells_ > 50_000:
            add("WARNING", f"PLANNER='jps' rebuilds a graph over all {int(cells_):,} cells on "
                           f"every replan (seconds per replan). Use PLANNER='local'.")
    else:
        add("ERROR", f"Unknown PLANNER={PLANNER!r} (use 'local' or 'jps').")
    if MPC_BACKEND not in ("parametric", "rebuild"):
        add("ERROR", f"Unknown MPC_BACKEND={MPC_BACKEND!r}.")
    if UNKNOWN_POLICY not in ("blocked", "optimistic"):
        add("ERROR", f"Unknown UNKNOWN_POLICY={UNKNOWN_POLICY!r}.")

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

    if len(TAR_WAYPOINTS) < 2:
        add("ERROR", "The target needs at least 2 waypoints.")
    elif os.environ.get("TARGET_MODE", "manual") == "manual":
        from target_manual import check_path
        _tp = np.array([w[:2] for w in TAR_WAYPOINTS])
        if TAR_SMOOTH_ENABLE and len(_tp) >= 3:
            from geometry import catmull_rom
            _tp = catmull_rom(_tp, TAR_SPLINE_DS)[0]
        _bad = check_path(_tp, TAR_RADIUS + SAFETY_MARGIN)
        if _bad:
            add("WARNING", f"Target path is closer than TAR_RADIUS + SAFETY_MARGIN = "
                           f"{TAR_RADIUS + SAFETY_MARGIN:.2f} m to an obstacle in {len(_bad)} "
                           f"place(s), worst clearance {min(b[2] for b in _bad):.2f} m. "
                           f"Fix with pick_waypoints.py.")
    if GENERATION_STATS is not None and not GENERATION_STATS["reached_density"]:
        add("WARNING", "map_gen could not reach the requested density "
                       "(raise max_tries or lower min_gap / density).")

    gap = _min_obstacle_gap(_rects, _circles)
    if gap is not None:
        passable = 2 * eff_inflate + GRID_RESOLUTION
        level = "INFO" if gap >= passable else "WARNING"
        add(level, f"Narrowest gap between obstacles: {gap:.2f} m "
                   f"(the grid planner needs about > {passable:.2f} m to pass through a gap).")
    return out


if __name__ == "__main__":
    print(f"Scenario {SCENARIO_NAME}: {NUM_ROBOT} UAV, world {WORLD_BOUNDS}, "
          f"horizon reach {VMAX * HORIZON_LENGTH * TIMESTEP:.2f} m")
    for level, msg in validate_config():
        print(f"[{level}] {msg}")
