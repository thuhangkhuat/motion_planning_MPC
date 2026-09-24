# Scenario files

One YAML file per scenario. `--scenario 6` loads `scen6.yaml`,
`--scenario big1000` loads `big1000.yaml`. Units are metres and m/s.

```yaml
description: free text

xlim: [0, 1000]             # map limits; also the occupancy-grid bounds
ylim: [0, 1000]
viewing_radius: 15          # half side of the square camera footprint
tar_max_speed: 5            # default target speed

starts:                     # UAV start positions [x, y, z] (z optional -> START_ALTITUDE)
- [40, 40, 30]

target:
  waypoints:                # [x, y] (z optional); the target passes through each point
  - [100, 100]
  - [450, 180]
  speeds: [5]               # optional, one value per segment (len = waypoints - 1)

obstacles:                  # fixed obstacles
  rects: [[x, y, w, h], ...]      # axis-aligned, (x, y) = lower-left corner
  circles: [[cx, cy, r], ...]

generate:                   # optional: add seeded random obstacles (map_gen.py)
  seed: 7
  density: 0.08             # fraction of the map area covered
  rect_size: [20, 60]       # side length range
  circle_radius: [10, 25]
  circle_fraction: 0.4      # share of circles among generated obstacles
  min_gap: 15               # clearance between any two obstacles (incl. fixed ones)
  border_margin: 10         # distance from the map border
  keep_clear_radius: 40     # free zone around UAV starts and target waypoints
  path_clearance: 12        # free corridor along the target path; 0 = off
  max_tries: 50000

params:                     # override any constant from config.py, e.g.
  VMAX: 10
  GRID_RESOLUTION: 2.0
```

## Picked points

`python pick_waypoints.py <scenario>` edits target waypoints (key `t`) and
UAV starts (key `u`) and saves them to `<name>.picks.json` next to the YAML
file. When that file exists its values replace `target` and `starts` from the
YAML. Delete it to go back to the YAML values.

## Generated maps

The same `generate` block always gives the same map: obstacles are placed
from the seed alone, and the keep-clear zones (starts, waypoints, target
path) are applied afterwards by removing the obstacles inside them. Moving a
start or a waypoint therefore only removes or restores obstacles around it.
Change `seed` for a different map.

    python map_gen.py big1000          # generation statistics
    python plot_scenario.py big1000 --traj
    python config.py big1000           # parameter consistency report

## Files

| File | |
|---|---|
| `scen1.yaml` … `scen9.yaml` | the original 50 m scenarios (converted from `config.py`; the old dict with its commented-out obstacles is in `legacy/config_scenarios_legacy.py`) |
| `big1000.yaml` | 1000 m example with a random map and scaled parameters |
