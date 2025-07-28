import numpy as np
import math
import heapq
from utils import *
import matplotlib.pyplot as plt
from lidar import *

class JumpPointSearch:
    def __init__(self):
        self.directions = [(-1, 0), ( 0,-1), ( 1, 0), ( 0, 1), 
                           (-1,-1), (-1, 1), ( 1,-1), ( 1, 1)]

    def plan(self, map:np.array, expanded_map:np.array,start:tuple, goal:tuple, htype=HeuristicType.MANHATTAN):
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
    

    
# ==============================================================================
# PHẦN 3: HÀM MAIN VÀ TRỰC QUAN HÓA
# ==============================================================================

def plot_result(map_array, path, start, goal):
    size_x, size_y = map_array.shape
    fig, ax = plt.subplots(figsize=(size_x, size_y))

    ax.imshow(map_array.T, cmap='Greys', origin='lower')

    ax.set_xticks(np.arange(-.5, map_array.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, map_array.shape[1], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, map_array.shape[0], 1))
    ax.set_yticks(np.arange(0, map_array.shape[1], 1))


    if path is not None and len(path) > 0:
        path_x = path[:, 0]
        path_y = path[:, 1]

        ax.plot(path_x, path_y, marker='o', color='orange', linestyle='-', linewidth=2, markersize=8, label='Jump Points Path')
    
  
    ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal')


    ax.legend()
    ax.set_title("Jump Point Search Pathfinding Result")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.show()

def createGridMap(data, pose, goal):
    size_x = int(2*max(SENSING_RADIUS, abs(goal[0]-pose[0]))/GRID_SIZE)+1
    size_y = int(2*max(SENSING_RADIUS, abs(goal[1]-pose[1]))/GRID_SIZE)+1

    print(size_x, size_y)
    grid_map = np.zeros((size_x, size_y))

    # Origin of the grid map
    origin_x = size_x // 2
    origin_y = size_y // 2

    # Convert polar to cartesian coordinates and update the grid map
    ang, dist = data
    # for angle, distance in lidar_data:
    for i in range(dist.shape[0]):
        angle = ang[i]; distance = dist[i]
        if distance > 0:  # avoid invalid measurements
            x = (distance-ROBOT_RADIUS) * np.cos(angle)
            y = (distance-ROBOT_RADIUS) * np.sin(angle)
            grid_x = int(origin_x + x / GRID_SIZE)
            grid_y = int(origin_y + y / GRID_SIZE)
            
            if 0 <= grid_x < size_x and 0 <= grid_y < size_y:
                grid_map[grid_x, grid_y] = 1

    # Start index
    start_idx = (origin_x, origin_y)
    goal_idx = (int(origin_x + (goal[0]-pose[0]) / GRID_SIZE),
                int(origin_y + (goal[1]-pose[1]) / GRID_SIZE))
    return grid_map, start_idx, goal_idx

def openingMap(grid_map):
    rows, cols = grid_map.shape
    mask = np.zeros((rows+2*EXPAND_SIZE, cols+2*EXPAND_SIZE))
    mask[EXPAND_SIZE:EXPAND_SIZE+rows, EXPAND_SIZE:EXPAND_SIZE+cols] = grid_map
    idxs, idys = np.where(grid_map>0)
    for i in range(idxs.shape[0]):
        mask[idxs[i]:idxs[i]+2*EXPAND_SIZE+1,
             idys[i]:idys[i]+2*EXPAND_SIZE+1] = np.ones((2*EXPAND_SIZE+1, 2*EXPAND_SIZE+1))
    grid_map = mask[EXPAND_SIZE:EXPAND_SIZE+rows, EXPAND_SIZE:EXPAND_SIZE+cols]
    return grid_map

if __name__ == "__main__":


    # --- Thiết lập kịch bản thử nghiệm ---
    
    # 1. Tạo bản đồ (0 = trống, 1 = chướng ngại vật)
    # Tọa độ được truy cập bằng (hàng, cột) hay (x, y)
    # grid_map = np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ], dtype=np.int8)

   
    
    # start_point = (3, 4) # (x, y)
    # goal_point = (5, 12) # (x, y)
    from robot import Robot
    pose = np.array([6.08356395, 4.71727118])
    goal = np.array([8.16776902,5.3322108])
    robots = [Robot(0, np.concatenate([[2.5, 0., 5., 0,0,0]]), np.zeros(3)),
              Robot(0, np.concatenate([[11., 6., 5., 0,0,0]]), np.zeros(3))]

    lidar = LidarScanner(range_min=0, range_max=SENSING_RADIUS,
                        angle_min=-math.pi, angle_max=math.pi, resolution=math.pi/45)
    import time
    st = time.time()
    data = lidar.senseObstacle(np.concatenate([pose, [0]]), robots)
    ang, dist = data
    print(time.time()-st)

    grid_map, start_idx, goal_idx = createGridMap(data, pose, goal)
    grid_map = openingMap(grid_map)

    # --- Chạy thuật toán và hiển thị kết quả ---
    
    print("="*30)
    print("Bắt đầu kiểm tra Jump Point Search")
    print(f"Bản đồ kích thước (X, Y): {grid_map.shape}")
    print(f"Bắt đầu từ: {start_idx}")
    print(f"Điểm đến: {goal_idx}")
    print("="*30)
    
    # 3. Khởi tạo và chạy JPS
    planner = JumpPointSearch()

    
    # Chú ý: JPS làm việc với (hàng, cột), nhưng ta plot theo (x, y)
    # Cần chuyển đổi tọa độ khi gọi hàm plan
    # start_idx = (start_point[1], start_point[0])
    # goal_idx = (goal_point[1], goal_point[0])
    
    path_indices = planner.plan(grid_map.T, start_idx, goal_idx)

    if path_indices is None:
        print("Không tìm thấy đường đi!")
        plot_result(grid_map, None, start_idx, goal_idx)
    else:
        # Chuyển đổi lại kết quả từ (hàng, cột) sang (x, y) để plot
        path_xy = np.array([(p[0], p[1]) for p in path_indices])
        
        print(f"Đã tìm thấy đường đi với {len(path_xy)} điểm nhảy.")
        print("Các điểm nhảy (Jump Points) theo tọa độ (x, y):")
        print(path_xy)
        
        # Gọi hàm plot
        plot_result(grid_map, path_xy, start_idx, goal_idx)