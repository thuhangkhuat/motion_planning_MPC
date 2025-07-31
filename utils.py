import math
import numpy as np
class HeuristicType:
    MANHATTAN = 1
    EUCLIDEAN = 2

class MapType:
    FREE = 0
    OBSTACLE = 1
    UNKNOWN = 2

def heuristic(start_node:tuple, end_node:tuple, htype:HeuristicType):
    if htype == HeuristicType.MANHATTAN:
        xdist = math.fabs(end_node[0] - start_node[0])
        ydist = math.fabs(end_node[1] - start_node[1])
        if xdist > ydist:
            return 14 * ydist + 10 * (xdist - ydist)
        else:
            return 14 * xdist + 10 * (ydist - xdist)
    elif htype == HeuristicType.EUCLIDEAN:
        return math.hypot(xdist, ydist)
    
def blocked(nx, ny, dx, dy, map:np.array):
    if nx + dx < 0 or nx + dx >= map.shape[1]:
        return True
    if ny + dy < 0 or ny + dy >= map.shape[0]:
        return True
    
    if map[ny + dy][nx + dx] == MapType.OBSTACLE:
        return True
    if dx != 0 and dy != 0:
        return map[ny][nx + dx] == MapType.OBSTACLE and \
               map[ny + dy][nx] == MapType.OBSTACLE
    elif dx != 0:
        return map[ny][nx + dx] == MapType.OBSTACLE
    else:
        return map[ny + dy][nx] == MapType.OBSTACLE

class Node:
    def __init__(self, n:list):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = np.inf


def is_collision(start:Node, end:Node, obstacle_points: np.array, robot_radius:float):
    if obstacle_points.shape[0] == 0:
        return False
    
    dx = end.x - start.x
    dy = end.y - start.y
    segment_length = math.hypot(dx, dy)

    if segment_length == 0:
        return False
    
    unit_dx = dx / segment_length
    unit_dy = dy / segment_length

    for point in obstacle_points:
        vec_start_to_point_x = point[0] - start.x
        vec_start_to_point_y = point[1] - start.y
        t = vec_start_to_point_x * unit_dx + vec_start_to_point_y * unit_dy
        closest_point_on_line_x = 0
        closest_point_on_line_y = 0
        if t <= 0:
            closest_point_on_line_x = start.x
            closest_point_on_line_y = start.y
        elif t >= segment_length:
            closest_point_on_line_x = end.x
            closest_point_on_line_y = end.y
        else:
            closest_point_on_line_x = start.x + t * unit_dx
            closest_point_on_line_y = start.y + t * unit_dy
        distance_to_segment = math.hypot(point[0] - closest_point_on_line_x, 
                                          point[1] - closest_point_on_line_y)
        if distance_to_segment <= robot_radius:
            return True
    return False


def get_ray(start:Node, end:Node):
    orig = [start.x, start.y]
    direc = [end.x - start.x, end.y - start.y]
    return orig, direc

def get_dist(start:Node, end:Node):
    return math.hypot(end.x - start.x, end.y - start.y)