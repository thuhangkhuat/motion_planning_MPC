import numpy as np
import math
import heapq
from utils import *

class JumpPointSearch:
    def __init__(self):
        self.directions = [(-1, 0), ( 0,-1), ( 1, 0), ( 0, 1), 
                           (-1,-1), (-1, 1), ( 1,-1), ( 1, 1)]
    
    

    def plan(self, map:np.array, expanded_map:np.array, start:tuple, goal:tuple, htype=HeuristicType.MANHATTAN):
        came_from = {}
        close_set = set()
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal, htype)}

        pqueue = []
        heapq.heappush(pqueue, (fscore[start], start))
        while pqueue:
            current = heapq.heappop(pqueue)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return np.array(path[::-1])

            close_set.add(current)
            
            for successor in self.get_successors(current, came_from, map, goal):
                jump_point = successor

                if (jump_point in close_set):
                    continue

                tentative_g_score = gscore[current] + self.get_length(current, jump_point, htype)
                if tentative_g_score < gscore.get(jump_point, 0) or jump_point not in [j[1] for j in pqueue]:
                    came_from[jump_point] = current
                    gscore[jump_point] = tentative_g_score
                    fscore[jump_point] = tentative_g_score + heuristic(jump_point, goal, htype)
                    heapq.heappush(pqueue, (fscore[jump_point], jump_point))
        return None

    def get_neighbors(self, node:tuple, parent:tuple, map:np.array):
        neighbors = []

        nx, ny = node
        if not isinstance(parent, tuple): # Check if parent is not a tuple
            for i, j in self.directions:
                if not blocked(nx, ny, i, j, map):
                    neighbors.append((nx + i, ny + j))
            return neighbors
        
        dx, dy = self.get_direction(node, parent)

        if dx != 0 and dy != 0:  # Diagonal movement
            if not blocked(nx, ny, 0, dy, map):
                neighbors.append((nx, ny + dy))
            if not blocked(nx, ny, dx, 0, map):
                neighbors.append((nx + dx, ny))
            if (not blocked(nx, ny, 0, dy, map) or not blocked(nx, ny, dx, 0, map)) and not blocked(nx, ny, dx, dy, map):
                neighbors.append((nx + dx, ny + dy))
            if blocked(nx, ny, -dx, 0, map) and not blocked(nx, ny, 0, dy, map):
                neighbors.append((nx - dx, ny + dy))
            if blocked(nx, ny, 0, -dy, map) and not blocked(nx, ny, dx, 0, map):
                neighbors.append((nx + dx, ny - dy))

        else:  # Straight movement
            if dx == 0:  # Moving vertically
                if not blocked(nx, ny, dx, 0, map):
                    if not blocked(nx, ny, 0, dy, map):
                        neighbors.append((nx, ny + dy))
                    if blocked(nx, ny, 1, 0, map):
                        neighbors.append((nx + 1, ny + dy))
                    if blocked(nx, ny, -1, 0, map):
                        neighbors.append((nx - 1, ny + dy))
            else:  # Moving horizontally
                if not blocked(nx, ny, dx, 0, map):
                    if not blocked(nx, ny, dx, 0, map):
                        neighbors.append((nx + dx, ny))
                    if blocked(nx, ny, 0, 1, map):
                        neighbors.append((nx + dx, ny + 1))
                    if blocked(nx, ny, 0, -1, map):
                        neighbors.append((nx + dx, ny - 1))

        return neighbors
    
    def jump(self, nx, ny, dx, dy, map:np.array, goal:tuple):
        newx = nx + dx
        newy = ny + dy

        if blocked(newx, newy, 0, 0, map):
            return None
        
        if (newx, newy) == goal:
            return newx, newy

        tx, ty = newx, newy

        if dx != 0 and dy != 0:  # Diagonal movement
            while True:
                if ((not blocked(tx, ty, -dx, dy, map) and blocked(tx, ty, -dx, 0, map)) or
                    (not blocked(tx, ty, dx, -dy, map) and blocked(tx, ty, 0, -dy, map))):
                    return tx, ty

                if self.jump(tx, ty, dx, 0, map, goal) or self.jump(tx, ty, 0, dy, map, goal):
                    return tx, ty

                tx += dx
                ty += dy

                if blocked(tx, ty, 0, 0, map) or self.dblock(tx, ty, dx, dy, map):
                    return None

                if (tx, ty) == goal:
                    return tx, ty

        else:  # Straight movement
            while True:
                if dx != 0:  # Horizontal movement
                    if ((not blocked(tx, newy, dx, 1, map) and blocked(tx, newy, 0, 1, map)) or
                        (not blocked(tx, newy, dx, -1, map) and blocked(tx, newy, 0, -1, map))):
                        return tx, newy

                    tx += dx
                    if blocked(tx, newy, 0, 0, map):
                        return None 
                    if (tx, newy) == goal:
                        return tx, newy

                else:  # Vertical movement
                    if ((not blocked(newx, ty, 1, dy, map) and blocked(newx, ty, 1, 0, map)) or
                        (not blocked(newx, ty, -1, dy, map) and blocked(newx, ty, -1, 0, map))):
                        return newx, ty

                    ty += dy
                    if blocked(newx, ty, 0, 0, map):
                        return None 
                    if (newx, ty) == goal:
                        return newx, ty
                        
    def get_successors(self, node:tuple, came_from:dict, map:np.array, goal:tuple):
        successors = []
        neighbors = self.get_neighbors(node, came_from.get(node, 0), map)
        nx, ny = node
        for cell in neighbors:
            cx, cy = cell
            jump_point = self.jump(nx, ny, cx - nx, cy - ny, map, goal)
            if jump_point is not None:
                successors.append(jump_point)   
        return successors
    
    def get_length(self, node:tuple, jump_point:tuple, htype:HeuristicType):
        dx, dy = self.get_direction(node, jump_point)
        lx, ly = abs(node[0] - jump_point[0]), abs(node[1] - jump_point[1])

        if htype == HeuristicType.MANHATTAN:
            if dx != 0 and dy != 0:
                return 14 * lx
            else:
                return 10 * (abs(dx) * lx + abs(dy) * ly)
        elif htype == HeuristicType.EUCLIDEAN:
            return math.hypot(lx, ly)
        
    @staticmethod
    def get_direction(node:tuple, parent:tuple):
        dx = np.sign(node[0] - parent[0])
        dy = np.sign(node[1] - parent[1])
        return (dx, dy)
    
    @staticmethod
    def dblock(nx, ny, dx, dy, map:np.array):
        return map[ny][nx - dx] == MapType.OBSTACLE and \
               map[ny - dy][nx] == MapType.OBSTACLE