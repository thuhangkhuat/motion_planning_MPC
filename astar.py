import math
import numpy as np
import matplotlib.pyplot as plt
from config import *
from lidar import *

class Node:
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index
    
class AStar:
    def __init__(self):
        self.motion = self.getMotion()

    def updatePlanner(self, grid_map, sidx, gidx):
        '''
        Initialize map for A Star planning
        grid_map: np.array - occupancy map
        sidx: (x, y) - start index
        gidx: (x, y) - goal index
        '''
        self.grid_map = grid_map
        self.sidx = sidx
        self.gidx = gidx
    
    def planning(self):
        """
        A star path search
        """
        start_node = Node(self.sidx[0], self.sidx[1], 0.0, -1)
        goal_node  = Node(self.gidx[0], self.gidx[1], 0.0, -1)
        open_set, close_set = dict(), dict()
        open_set[self.getGridIndex(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                break
            
            c_id = min(open_set,
                    key=lambda o:open_set[o].cost + self.getHeuristic(goal_node, open_set[o]))
            
            current = open_set[c_id]
                
            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            close_set[c_id] = current

            for motion in self.motion:
                node = Node(current.x + motion[0],
                            current.y + motion[1],
                            current.cost + motion[2], c_id)
                n_id = self.getGridIndex(node)

                if not self.isValid(node):
                    continue

                if n_id in close_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node
        
        rx, ry = self.getFinalGridPath(goal_node, close_set)
        
        # reset
        self.grid_map = None
        self.sidx = None
        self.gidx = None
        return rx, ry

    def getFinalGridPath(self, goal_node, closed_set):
        # generate final course
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x), ry.append(n.y)
            parent_index = n.parent_index

        return rx, ry

    def getGridIndex(self, node):
        return node.y * self.grid_map.shape[0] + node.x

    @staticmethod
    def getHeuristic(n1, n2):
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def isValid(self, node):
        if node.x < 0:
            return False
        elif node.y < 0:
            return False
        elif node.x >= self.grid_map.shape[0]:
            return False
        elif node.y >= self.grid_map.shape[1]:
            return False

        # collision check
        if self.grid_map[node.x][node.y] > 0:
            return False

        return True

    @staticmethod
    def getMotion():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, 1],
                  [-1, 1, 1],
                  [1, -1, 1],
                  [1, 1, 1]]

        return motion
    
