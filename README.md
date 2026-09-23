# motion_planning_MPC

Multi-UAV target tracking in cluttered environments: NMPC (CasADi/IPOPT) with a
CBF visibility constraint for the leader, safe corridors (pydecomp) built from
LiDAR scans, JPS on a log-odds occupancy grid, and a two-phase SEARCH / TRACK
scheme with a satellite formation.

## Installation (Python 3.12)

`pydecomp 2.0.0` requires `pycddlib` **2.x** (3.x is not compatible), and
`pycddlib` needs the GMP library to build. Install in this order:

```bash
# 1) System library for pycddlib
#    Ubuntu/WSL:  sudo apt install libgmp-dev
#    macOS:       brew install gmp
#    conda (any OS, no sudo):  conda install -c conda-forge gmp
#    Windows (conda): if the build fails -> conda install -c conda-forge "pycddlib<3"

# 2) Python packages
pip install -r requirements.txt
```

Check: `python -c "import casadi, pathfind, pydecomp; print('ok')"`

## Running

```bash
python main.py --scenario 6                      # run until the target reaches its goal
python main.py --scenario 6 --max-steps 300      # short test run
python main.py --scenario 6 --seed 2 --quiet     # different seed, minimal console output
python main.py -h                                # all options
```

Each run creates `runs/scen<N>_n<num UAVs>_s<seed>_<timestamp>/`:

| File | Contents |
|---|---|
| `data.pkl` | UAV and target trajectories, corridors, compute times (same format as before) |
| `config.json` | all parameters, scenario, target waypoints, seed, git commit |
| `log.txt` | full log (MPC infeasibility, JPS failures, emergency stops, ...) |

- A checkpoint is written every 500 steps (`--checkpoint-every`).
- Ctrl+C stops after the current step and still saves the data (press twice to stop immediately).
- The same scenario + seed + code gives bit-identical results.

## Configuration

All tunable values live in `config.py`:

1. **Common parameters** — dynamics, LiDAR, occupancy grid (`GRID_RESOLUTION`,
   `INFLATE_MARGIN`, `WORLD_BOUNDS`), replanning thresholds, Kalman / lead
   pursuit, MPC weights, IPOPT options, formation parameters.
2. **Scenarios** — obstacles, UAV starts, target waypoints, map limits.
   Any common parameter can be overridden per scenario in `params`.
3. **Derived parameters** — values tied to others (`INFLATE_RADIUS`,
   `COLLISION_AVOID_DISTANCE`, `FORMATION_GAP`, `STANDOFF_DISTANCE`,
   `CORRIDOR_BOX`, `WORLD_BOUNDS`, ...) are computed after the scenario is
   loaded, so they follow its `viewing_radius` / speeds. Put the name in
   `params` to pin a value instead.
4. **Consistency checks** — `validate_config()` flags coupled parameters that
   no longer fit together (horizon reach vs corridor box and neighbour range,
   LiDAR ray spacing vs grid cell, UAVs tunnelling between samples, starts or
   waypoints outside the world, narrowest obstacle gap vs grid, ...).

```bash
python config.py                   # report for the scenario in SCENARIO
SCENARIO=6 python config.py
```

`main.py` logs the same report at start-up and refuses to run if it contains
an `ERROR` (override with `--ignore-config-errors`).

## Target: picking waypoints

```bash
python pick_waypoints.py 6         # click to place points, Enter to save
python plot_scenario.py 6 --traj   # preview the target trajectory
```

Waypoints are stored in `scenarios/target_scen<N>.json`; an optional `"speeds"`
list sets the speed of each segment. If the file does not exist, the
`waypoints` entry in `config.py` is used. The trajectory passes exactly through
the chosen points (centripetal Catmull-Rom spline when `TAR_SMOOTH_ENABLE` is
on), and segments that come too close to obstacles are reported as warnings.

The previous RRT-based target: `python main.py --target-mode rrt`.

## Plotting results

```bash
RUN_FILE=runs/<run>/data.pkl SCENARIO=6 python plot_result.py
RUN_FILE=runs/<run>/data.pkl SCENARIO=6 python plot_animation.py
```

Without `RUN_FILE`, the plot scripts read the legacy path
(`data<M>_scen<N>_<n>.txt`); `main.py` still writes a copy there on every run
(disable with `--no-legacy-copy`).

## Project layout

| File | Role |
|---|---|
| `main.py` | simulation loop, CLI, result saving |
| `config.py` | shared parameters + scenario definitions |
| `robot_jps.py` | UAV: NMPC, CBF, SEARCH/TRACK, formation slots, path commit |
| `planner_jps.py` | log-odds occupancy grid + JPS |
| `lidar.py` | simulated LiDAR (analytic for rectangles, ray-marching for circles) |
| `kalman_target.py` | Kalman filter for target velocity (lead pursuit) |
| `target_manual.py`, `pick_waypoints.py` | target trajectory from user-picked waypoints |
| `target_rrt.py`, `target_cache.py` | RRT-based target trajectory (legacy mode) |
| `plot_*.py`, `metrics.py`, `collect_metrics.py` | plotting and statistics |
| `legacy/` | old versions, no longer imported |