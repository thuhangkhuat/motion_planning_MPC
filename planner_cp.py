"""
Debug script: so sánh A* vs JPS trên cùng setup để xác định bug.

Chạy:
    python debug_jps.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Tăng recursion limit để loại trừ hypothesis 1
sys.setrecursionlimit(50000)

from config import *
from planner_astar import AStarPlanner, OccupancyGrid
from planner_jps import JPSPlanner


def test_case(name, start, goal, obstacle_points):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  Start: {start}, Goal: {goal}")
    print(f"  Obstacles: {len(obstacle_points)} points")
    print(f"{'='*60}")
    
    # ---- A* ----
    print("\n[A*]")
    astar = AStarPlanner()
    astar.initialize(start, goal)
    success_a, path_a, exp_a, _, iters_a = astar.plan(
        obstacle_points, ROBOT_RADIUS, time_budget_ms=500)
    print(f"  success={success_a}, iters={iters_a}, path_len={len(path_a)}")
    if astar.was_goal_projected:
        print(f"  Goal projected to: {astar.projected_goal}")
    
    # ---- JPS ----
    print("\n[JPS]")
    jps = JPSPlanner()
    jps.initialize(start, goal)
    try:
        success_j, path_j, exp_j, _, iters_j = jps.plan(
            obstacle_points, ROBOT_RADIUS, time_budget_ms=500)
        print(f"  success={success_j}, iters={iters_j}, path_len={len(path_j)}")
        if jps.was_goal_projected:
            print(f"  Goal projected to: {jps.projected_goal}")
    except RecursionError as e:
        print(f"  RECURSION ERROR: {e}")
        success_j = False
        path_j = []
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        success_j = False
        path_j = []
    
    # ---- Verdict ----
    print("\n[VERDICT]")
    if success_a and success_j:
        print("  Both OK")
    elif success_a and not success_j:
        print("  *** A* OK but JPS FAILED — bug in JPS! ***")
    elif not success_a and success_j:
        print("  Weird: JPS OK but A* failed")
    else:
        print("  Both failed (probably unreachable)")
    
    return success_a, success_j, path_a, path_j


if __name__ == "__main__":
    # Test 1: Empty map, simple goal
    obs_empty = np.array([]).reshape(0, 2)
    test_case("Empty map, short distance",
              start=(5.0, 5.0),
              goal=(10.0, 10.0),
              obstacle_points=obs_empty)
    
    # Test 2: Empty map, long distance
    test_case("Empty map, long diagonal",
              start=(0.0, 0.0),
              goal=(40.0, 40.0),
              obstacle_points=obs_empty)
    
    # Test 3: Single obstacle in between
    obs_single = np.array([
        [x, y] for x in np.arange(8, 12, 0.5) for y in np.arange(8, 12, 0.5)
    ])
    test_case("Single obstacle",
              start=(5.0, 5.0),
              goal=(15.0, 15.0),
              obstacle_points=obs_single)
    
    # Test 4: Scenario 5 starts (if config loaded)
    try:
        obs_scenario = np.array([])
        # Try to create some obstacle points from RECTANGLE_OBSTACLES
        pts = []
        for rect in RECTANGLE_OBSTACLES:
            x, y, w, h = rect
            # Sample boundary points
            for t in np.arange(0, 1, 0.1):
                pts.append([x + t*w, y])
                pts.append([x + t*w, y + h])
                pts.append([x, y + t*h])
                pts.append([x + w, y + t*h])
        obs_scenario = np.array(pts)
        
        # Test với scenario 5 start
        test_case("Scenario 5 (robot 0)",
                  start=(STARTS[0][0], STARTS[0][1]),
                  goal=(GOALS[0][0], GOALS[0][1]),
                  obstacle_points=obs_scenario)
    except Exception as e:
        print(f"\nSkipped scenario test: {e}")
    
    # Test 5: Goal at exact start (edge case)
    test_case("Start = Goal",
              start=(5.0, 5.0),
              goal=(5.0, 5.0),
              obstacle_points=obs_empty)
    
    print("\n" + "="*60)
    print("Done. Tìm test case mà A* OK nhưng JPS fail.")
    print("="*60)