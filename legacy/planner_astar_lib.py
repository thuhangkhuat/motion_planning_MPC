"""
A* planner via library `pathfind` — chỉ dùng cho FAIR COMPARISON với JPS.

KHÔNG dùng làm main planner. Main planner là astar_planner.AStarPlanner
(custom numpy implementation, nhanh hơn).

Mục đích: so sánh A* vs JPS với CÙNG library overhead.
Cả 2 planners dùng pathfind library → diff time = diff algorithm.

API tương thích AStarPlanner:
    planner = AStarLibPlanner()
    planner.initialize(start, goal)
    success, path, _, _, iters = planner.plan(obstacle_points, robot_radius,
                                              time_budget_ms=30)
"""

import time
import numpy as np

from config import *
from planner_astar import (
    OccupancyGrid,
    project_goal_to_grid,
    remove_residual_node,
    ASTAR_GRID_RESOLUTION,
    ASTAR_INFLATE_RADIUS,
    ASTAR_LOCAL_HALF_SIZE,
    _nearest_free,
)

try:
    import pathfind
    PATHFIND_AVAILABLE = True
except ImportError:
    PATHFIND_AVAILABLE = False
    print("WARNING: 'pathfind' library not installed. Run: pip install pathfind")


def _occupancy_to_matrix(occ_grid: OccupancyGrid):
    """Convert OccupancyGrid sang pathfind matrix format."""
    matrix = np.where(occ_grid.grid > 0, -1, 1).tolist()
    return matrix


class AStarLibPlanner:
    """
    A* via pathfind library.
    
    Mục đích DUY NHẤT: fair comparison với JPSPlanner (cùng library).
    KHÔNG dùng làm main planner trong robot pipeline.
    """
    
    def __init__(self,
                 grid_resolution=ASTAR_GRID_RESOLUTION,
                 inflate_radius=ASTAR_INFLATE_RADIUS,
                 local_half_size=ASTAR_LOCAL_HALF_SIZE):
        if not PATHFIND_AVAILABLE:
            raise ImportError(
                "Library 'pathfind' chưa được cài. "
                "Chạy: pip install pathfind")
        
        self.grid_resolution = grid_resolution
        self.inflate_radius = inflate_radius
        self.local_half_size = local_half_size
        
        self.start = None
        self.goal = None
        self.grid = None
        self.expanded = set()
        self.was_goal_projected = False
        self.projected_goal = None
    
    def initialize(self, start, goal):
        self.start = np.asarray(start[:2], dtype=float)
        self.goal = np.asarray(goal[:2], dtype=float)
    
    def plan(self, obstacle_points, robot_radius,
             max_iter=None, time_budget_ms=None):
        if self.start is None or self.goal is None:
            return False, [], set(), set(), 0
        
        # Build local OccupancyGrid (giống các planner khác)
        self.grid = OccupancyGrid(
            center=self.start,
            half_size=self.local_half_size,
            obstacle_points=obstacle_points,
            resolution=self.grid_resolution,
            inflate_radius=max(self.inflate_radius, robot_radius))
        
        # Goal projection
        local_goal, was_projected = project_goal_to_grid(
            self.start, self.goal, self.grid)
        self.was_goal_projected = was_projected
        self.projected_goal = local_goal
        
        sx, sy = self.grid.world_to_grid(self.start[0], self.start[1])
        gx, gy = self.grid.world_to_grid(local_goal[0], local_goal[1])
        
        if not self.grid.in_bounds(sx, sy) or not self.grid.in_bounds(gx, gy):
            return False, [], set(), set(), 0
        
        if not self.grid.is_free(sx, sy):
            result = _nearest_free(self.grid, sx, sy)
            if result is None or result[0] is None:
                return False, [], set(), set(), 0
            sx, sy = result
        if not self.grid.is_free(gx, gy):
            result = _nearest_free(self.grid, gx, gy)
            if result is None or result[0] is None:
                return False, [], set(), set(), 0
            gx, gy = result
        
        # Edge case: start == goal
        if (sx, sy) == (gx, gy):
            wx, wy = self.grid.grid_to_world(sx, sy)
            return True, [[wx, wy]], set(), set(), 1
        
        matrix = _occupancy_to_matrix(self.grid)
        
        start_str = f"{sy},{sx}"
        end_str = f"{gy},{gx}"
        
        t_start = time.time()
        try:
            graph = pathfind.transform.matrix2graph(matrix, diagonal=True)
            # Khác với JPSPlanner: dùng method="a*" thay vì "jps"
            path_strs = pathfind.find(graph, start=start_str, end=end_str,
                                       method="a*")
        except Exception as e:
            print(f"A* lib error: {type(e).__name__}: {e}")
            return False, [], set(), set(), 0
        
        elapsed_ms = (time.time() - t_start) * 1000
        
        if not path_strs:
            return False, [], set(), set(), 0
        
        path_world = []
        for s in path_strs:
            parts = s.split(",")
            iy = int(parts[0])
            ix = int(parts[1])
            wx, wy = self.grid.grid_to_world(ix, iy)
            path_world.append([wx, wy])
        
        self.expanded = set()
        return True, path_world, self.expanded, set(), len(path_strs)