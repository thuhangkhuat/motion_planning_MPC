"""
Jump Point Search (JPS) planner — wrapper quanh library `pathfind`.

Yêu cầu install:
    pip install pathfind

Library: https://github.com/MorvanZhou/pathfind
API:
    graph = pathfind.transform.matrix2graph(matrix, diagonal=True)
    path = pathfind.find(graph, start="row,col", end="row,col", method="jps")

Matrix value convention:
    -1 = obstacle (infinity cost)
    >= 1 = walkable (cost = value)

Note: pathfind dùng "row,col" coordinate system (matrix index),
khác với pathfinding (x, y). Cần convert cẩn thận.

API tương thích AStarPlanner:
    planner = JPSPlanner()
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
    """
    Convert OccupancyGrid sang matrix format của pathfind.
    
    OccupancyGrid: numpy array (rows=size_y, cols=size_x)
        0 = free, 1 = obstacle
    
    pathfind matrix: list of list
        -1 = obstacle, >=1 = walkable
    
    Returns: matrix (list of list)
    """
    # Convert: 0 -> 1 (walkable), 1 -> -1 (obstacle)
    matrix = np.where(occ_grid.grid > 0, -1, 1).tolist()
    return matrix


class JPSPlanner:
    """
    JPS planner dùng library `pathfind`.
    Drop-in replacement cho AStarPlanner.
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
        
        # Build local OccupancyGrid (giống A*)
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
        
        # Grid coords
        sx, sy = self.grid.world_to_grid(self.start[0], self.start[1])
        gx, gy = self.grid.world_to_grid(local_goal[0], local_goal[1])
        
        # Validate bounds
        if not self.grid.in_bounds(sx, sy) or not self.grid.in_bounds(gx, gy):
            return False, [], set(), set(), 0
        
        # Nearest-free fallback
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
        
        # Convert OccupancyGrid sang pathfind matrix
        matrix = _occupancy_to_matrix(self.grid)
        
        # pathfind dùng (row, col) - row tương ứng y, col tương ứng x trong grid
        # OccupancyGrid: grid[iy, ix] -> matrix[iy][ix]
        # Vậy: row = iy = sy, col = ix = sx
        start_str = f"{sy},{sx}"
        end_str = f"{gy},{gx}"
        
        t_start = time.time()
        try:
            graph = pathfind.transform.matrix2graph(matrix, diagonal=True)
            path_strs = pathfind.find(graph, start=start_str, end=end_str,
                                       method="jps")
        except Exception as e:
            print(f"JPS library error: {type(e).__name__}: {e}")
            return False, [], set(), set(), 0
        
        elapsed_ms = (time.time() - t_start) * 1000
        
        if not path_strs:
            return False, [], set(), set(), 0
        
        # Convert path strings về world coords
        # Path strings: ["row,col", ...] = ["iy,ix", ...]
        path_world = []
        for s in path_strs:
            parts = s.split(",")
            iy = int(parts[0])
            ix = int(parts[1])
            wx, wy = self.grid.grid_to_world(ix, iy)
            path_world.append([wx, wy])
        
        # pathfind không expose expanded set
        self.expanded = set()
        
        return True, path_world, self.expanded, set(), len(path_strs)