import numpy as np
import math

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
