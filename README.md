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
2. **Scenarios** — loaded from `scenarios/*.yaml` (obstacles, UAV starts,
   target waypoints, map limits, optional random map generation). Any common
   parameter can be overridden per scenario in `params`. The file format is
   described in [`scenarios/README.md`](scenarios/README.md).
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
python config.py 6                 # report for scenarios/scen6.yaml
python config.py big1000
```

`main.py` logs the same report at start-up and refuses to run if it contains
an `ERROR` (override with `--ignore-config-errors`).

## Performance-related options

| Option (config.py / scenario `params`) | Values | |
|---|---|---|
| `MPC_BACKEND` | `parametric` (default), `rebuild` | `parametric` builds the NMPC once per UAV (`mpc_problem.py`) and only updates parameters each step; `rebuild` is the original per-step `casadi.Opti` |
| `PLANNER` | `local` (default), `jps` | `local` = 8-connected Dijkstra (numba) in a window of half size `LOCAL_PLAN_RADIUS` around the UAV (`planner_grid.py`); `jps` = original `pathfind` JPS over the whole grid |
| `UNKNOWN_POLICY` | `blocked` (default), `optimistic` | whether never-observed cells may be planned through (`optimistic` is needed on large maps; `UNKNOWN_COST` > 1 prefers known free space) |
| `LIDAR_MODE` | `analytic` (default), `march` | exact vectorised ray casting vs. the original ray marching |
| `COST_LENGTH_SCALE`, `COST_ACCEL_SCALE` | 1.0 (default) | distances / controls are divided by these before entering the MPC cost, so weights keep their meaning on larger maps; 1.0 reproduces the original cost |

**Failsafe.** When the planner fails `MAX_FAIL_BEFORE_STOP` times in a row
(emergency stop) or the MPC fails more than `MPC_FAILS_BEFORE_BRAKE` times in a
row, the UAV brakes at up to `UMAX` (`FAILSAFE_BRAKE: true`, default). The
original behaviour applied zero acceleration, which keeps the current velocity,
and replayed a stale plan on repeated MPC failures; both let UAVs drift into
obstacles. `FAILSAFE_BRAKE: false` restores it.

The original implementations are kept so results can be compared:
`MPC_BACKEND: rebuild`, `PLANNER: jps`, `LIDAR_MODE: march` in a scenario's
`params` restores the previous pipeline.

## Scenarios, target waypoints and UAV starts

```bash
python pick_waypoints.py 6         # t = target waypoints, u = UAV starts, Enter = save
python plot_scenario.py 6 --traj   # preview the map and the target trajectory
python map_gen.py big1000          # statistics of a generated map
```

Picked points are stored in `scenarios/<name>.picks.json` and override the
YAML values; an optional `"speeds"` list sets the speed of each target
segment. The trajectory passes exactly through the chosen points (centripetal
Catmull-Rom spline when `TAR_SMOOTH_ENABLE` is on), and segments that come
too close to obstacles are reported as warnings.

Large maps: add a `generate:` block to the scenario file to place seeded
random obstacles (density, size ranges, minimum gap, keep-clear zones around
starts, waypoints and the target path). `scenarios/big1000.yaml` is a
1000 m × 1000 m example with scaled parameters.

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
| `config.py` | shared parameters, scenario loading, derived parameters, consistency checks |
| `scenarios/` | scenario files (`*.yaml`) and picked points (`*.picks.json`) |
| `map_gen.py` | seeded random obstacle maps |
| `geometry.py` | spline / resampling helpers |
| `robot_jps.py` | UAV: NMPC, CBF, SEARCH/TRACK, formation slots, path commit |
| `planner_grid.py` | local-window grid planner (numba Dijkstra, incremental grid rebuild) |
| `planner_jps.py` | log-odds occupancy grid + JPS (original planner) |
| `mpc_problem.py` | parametric NMPC, built once per UAV |
| `lidar.py` | simulated LiDAR (vectorised exact ray casting; original ray marching kept) |
| `kalman_target.py` | Kalman filter for target velocity (lead pursuit) |
| `target_manual.py` | target trajectory through the chosen waypoints |
| `pick_waypoints.py` | interactive editor for target waypoints and UAV starts |
| `target_rrt.py`, `target_cache.py` | RRT-based target trajectory (legacy mode) |
| `plot_*.py`, `metrics.py`, `collect_metrics.py` | plotting and statistics |
| `legacy/` | old versions, no longer imported |