import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon # Import để vẽ đa giác
from matplotlib.path import Path

from config import *

class LidarScanner:
    def __init__(self, range_min=0.1, range_max=100.0,
                            angle_min=-math.pi/2, angle_max = math.pi/2,
                            resolution=math.pi/90, noise=0.01):
        self.range_min = range_min
        self.range_max = range_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.resolution = resolution
        self.angle_num = int((self.angle_max - self.angle_min)/self.resolution) + 1
        self.range_num = int(10*range_max)
        self.noise = noise

    def distance(self, pose, obs_pose):
        ex = obs_pose[0] - pose[0]
        ey = obs_pose[1] - pose[1]
        return math.hypot(ex, ey)
    
    def is_point_on_segment(self, p, a, b):
        return np.cross(p - a, b - a) == 0 and \
               min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and \
               min(a[1], b[1]) <= p[1] <= max(a[1], b[1])

    def do_segments_intersect(self, p1, q1, p2, q2):
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - \
                  (q[0] - p[0]) * (r[1] - q[1])
            if val == 0: return 0 
            return 1 if val > 0 else 2 

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0 and \
           o1 != o2 and o3 != o4: # Check if points are on opposite sides
            return True

        if o1 == 0 and self.is_point_on_segment(p2, p1, q1): return True
        if o2 == 0 and self.is_point_on_segment(q2, p1, q1): return True
        if o3 == 0 and self.is_point_on_segment(p1, p2, q2): return True
        if o4 == 0 and self.is_point_on_segment(q1, p2, q2): return True

        return False

    def get_intersection_point(self, p1, q1, p2, q2):
        A1 = q1[1] - p1[1]
        B1 = p1[0] - q1[0]
        C1 = A1 * p1[0] + B1 * p1[1]

        A2 = q2[1] - p2[1]
        B2 = p2[0] - q2[0]
        C2 = A2 * p2[0] + B2 * p2[1]

        determinant = A1 * B2 - A2 * B1

        if determinant == 0:
            return None 
        else:
            x = (B2 * C1 - B1 * C2) / determinant
            y = (A1 * C2 - A2 * C1) / determinant
            epsilon = 1e-9
            if (min(p1[0], q1[0]) - epsilon <= x <= max(p1[0], q1[0]) + epsilon) and \
               (min(p1[1], q1[1]) - epsilon <= y <= max(p1[1], q1[1]) + epsilon) and \
               (min(p2[0], q2[0]) - epsilon <= x <= max(p2[0], q2[0]) + epsilon) and \
               (min(p2[1], q2[1]) - epsilon <= y <= max(p2[1], q2[1]) + epsilon):
                return np.array([x, y])
            return None

    def isCollision(self, pose, obstacles):    
        for (cx, cy, cr) in obstacles:
            ex = cx - pose[0]; ey = cy - pose[1]
            if ex**2 + ey**2 - cr**2 <= 0:
                return True
        return False

    def senseObstacle(self, pose, robots):
        circle_obstacles_in_range = []
        polygon_obstacles_in_range = []

        for i in range(OBSTACLES.shape[0]):
            # Mở rộng bán kính vật cản bằng bán kính robot để tránh va chạm
            if np.hypot(pose[0]-OBSTACLES[i,0],
                        pose[1]-OBSTACLES[i,1]) < SENSING_RADIUS + OBSTACLES[i,2] + ROBOT_RADIUS:
                circle_obstacles_in_range.append(OBSTACLES[i,:])
        
        # Lấy danh sách các vật cản đa giác trong tầm quét
        # (Bạn có thể tinh chỉnh điều kiện này để chỉ bao gồm các đa giác thực sự gần)
        for poly in POLYGON_OBSTACLES:
            polygon_obstacles_in_range.append(poly)


        data = [] # Lưu trữ các khoảng cách phát hiện được
        x_robot, y_robot, theta_robot = pose # Vị trí và hướng của robot
        
        # Lặp qua từng góc quét của LiDAR
        for angle_offset in np.linspace(self.angle_min, self.angle_max, self.angle_num, True):
            current_angle = theta_robot + angle_offset # Góc tuyệt đối của tia LiDAR
            
            # Định nghĩa tia LiDAR dưới dạng một đoạn thẳng từ robot đến giới hạn tầm quét
            p1_ray = np.array([x_robot, y_robot])
            p2_ray = np.array([x_robot + self.range_max * math.cos(current_angle),
                               y_robot + self.range_max * math.sin(current_angle)])

            min_dist_along_ray = self.range_max # Khoảng cách nhỏ nhất phát hiện được cho tia này

            # --- KIỂM TRA VA CHẠM VỚI VẬT CẢN HÌNH TRÒN ---
            for (cx, cy, cr) in circle_obstacles_in_range:
                # Bán kính hiệu quả để tính toán va chạm (vật cản + robot_radius)
                effective_radius = cr + ROBOT_RADIUS 

                # Vector từ robot đến tâm vật cản
                v_rc = np.array([cx - x_robot, cy - y_robot])
                dist_rc_sq = np.dot(v_rc, v_rc) # Bình phương khoảng cách từ robot đến tâm vật cản

                # Vector hướng của tia LiDAR (chuẩn hóa)
                v_ray_dir = np.array([math.cos(current_angle), math.sin(current_angle)])

                # Chiếu vector v_rc lên v_ray_dir
                proj_dist_along_ray = np.dot(v_rc, v_ray_dir)

                # Khoảng cách vuông góc từ tâm vật cản đến đường thẳng chứa tia LiDAR
                dist_perp_sq = dist_rc_sq - proj_dist_along_ray**2

                # Nếu khoảng cách vuông góc nhỏ hơn bình phương bán kính hiệu quả, có thể có giao điểm
                if dist_perp_sq < effective_radius**2:
                    # Tính toán khoảng cách từ điểm gần nhất trên tia đến tâm vật cản theo hướng vuông góc
                    dist_to_tangent = math.sqrt(effective_radius**2 - dist_perp_sq)
                    
                    # Hai điểm giao trên tia (tính từ robot)
                    d1 = proj_dist_along_ray - dist_to_tangent
                    d2 = proj_dist_along_ray + dist_to_tangent

                    # Chỉ xem xét các giao điểm nằm phía trước robot và trong tầm quét
                    if d1 > self.range_min and d1 < self.range_max:
                        min_dist_along_ray = min(min_dist_along_ray, d1)
                    if d2 > self.range_min and d2 < self.range_max:
                        min_dist_along_ray = min(min_dist_along_ray, d2)
                
                # Xử lý trường hợp robot bắt đầu từ bên trong vật cản hình tròn
                if dist_rc_sq < effective_radius**2:
                    min_dist_along_ray = min(min_dist_along_ray, self.range_min)


            # --- KIỂM TRA VA CHẠM VỚI VẬT CẢN ĐA GIÁC ---
            for poly in polygon_obstacles_in_range:
                num_vertices = len(poly)
                for i in range(num_vertices):
                    p3_segment = poly[i]
                    p4_segment = poly[(i + 1) % num_vertices]

                    intersection_point = self.get_intersection_point(p1_ray, p2_ray, p3_segment, p4_segment)
                    
                    if intersection_point is not None:
                        dist_to_intersection = self.distance(pose, intersection_point)
                        # Chỉ xem xét các giao điểm nằm phía trước robot và trong tầm quét
                        if dist_to_intersection > self.range_min and dist_to_intersection < self.range_max: 
                            min_dist_along_ray = min(min_dist_along_ray, dist_to_intersection)
            
            data.append(min_dist_along_ray + np.random.rand() * self.noise) # Thêm nhiễu
        
        angle = np.linspace(self.angle_min, self.angle_max, self.angle_num)
        data = np.array(data)

        idx = np.where(data < self.range_max) # Lọc bỏ các điểm không có vật cản
        angle = angle[idx]
        data = data[idx]
        return angle, data
    
    def getObstaclePoints(self, data, pose):
        scan_angles, scan_ranges = data
        valid_indices = scan_ranges < self.range_max
        
        detected_angles = scan_angles[valid_indices]
        detected_ranges = scan_ranges[valid_indices]
        
        obstacle_points = np.vstack([
            pose[0] + detected_ranges * np.cos(detected_angles),
            pose[1] + detected_ranges * np.sin(detected_angles)]).T
        
        return obstacle_points

def getCircle(x,y,r):
    theta = np.linspace(0, 2*np.pi, 50)   
    a = x + r * np.cos(theta)
    b = y + r * np.sin(theta)
    return a, b

def createGridMap(data, pose, goal):

    
    # Giới hạn cơ sở từ config
    grid_min_x_world = XLIM[0]
    grid_max_x_world = XLIM[1]
    grid_min_y_world = YLIM[0]
    grid_max_y_world = YLIM[1]

    # Mở rộng giới hạn nếu robot, tầm quét, hoặc mục tiêu nằm ngoài
    grid_min_x_world = min(grid_min_x_world, pose[0] - SENSING_RADIUS - ROBOT_RADIUS)
    grid_max_x_world = max(grid_max_x_world, pose[0] + SENSING_RADIUS + ROBOT_RADIUS)
    grid_min_y_world = min(grid_min_y_world, pose[1] - SENSING_RADIUS - ROBOT_RADIUS)
    grid_max_y_world = max(grid_max_y_world, pose[1] + SENSING_RADIUS + ROBOT_RADIUS)

    grid_min_x_world = min(grid_min_x_world, goal[0] - ROBOT_RADIUS)
    grid_max_x_world = max(grid_max_x_world, goal[0] + ROBOT_RADIUS)
    grid_min_y_world = min(grid_min_y_world, goal[1] - ROBOT_RADIUS)
    grid_max_y_world = max(grid_max_y_world, goal[1] + ROBOT_RADIUS)


    size_x_world = grid_max_x_world - grid_min_x_world
    size_y_world = grid_max_y_world - grid_min_y_world

    size_x_grid = int(size_x_world / GRID_SIZE) + 1
    size_y_grid = int(size_y_world / GRID_SIZE) + 1

    grid_map = np.zeros((size_x_grid, size_y_grid))

    # Xử lý các điểm phát hiện bởi LiDAR
    ang, dist = data
    for i in range(dist.shape[0]):
        angle_relative_to_robot = ang[i]
        distance_to_obstacle = dist[i]

        if distance_to_obstacle > 0:
            # Chuyển đổi sang tọa độ Cartesian toàn cầu (tâm vật cản, chưa mở rộng)
            global_angle = pose[2] + angle_relative_to_robot 
            ox_global = pose[0] + distance_to_obstacle * np.cos(global_angle)
            oy_global = pose[1] + distance_to_obstacle * np.sin(global_angle)
            
            # Chuyển đổi sang tọa độ lưới
            grid_x = int((ox_global - grid_min_x_world) / GRID_SIZE)
            grid_y = int((oy_global - grid_min_y_world) / GRID_SIZE)
            
            if 0 <= grid_x < size_x_grid and 0 <= grid_y < size_y_grid:
                grid_map[grid_x, grid_y] = 1 # Đánh dấu là vật cản

    # Đánh dấu vật cản hình tròn trực tiếp lên lưới (đã mở rộng bán kính robot)
    for (cx, cy, cr) in OBSTACLES:
        effective_radius = cr + ROBOT_RADIUS # Bán kính vật cản hiệu quả
        for gx in range(size_x_grid):
            for gy in range(size_y_grid):
                world_x = grid_min_x_world + gx * GRID_SIZE
                world_y = grid_min_y_world + gy * GRID_SIZE
                if np.hypot(world_x - cx, world_y - cy) < effective_radius:
                    grid_map[gx, gy] = 1

    # Đánh dấu vật cản đa giác trực tiếp lên lưới (đã mở rộng bán kính robot)
    for poly in POLYGON_OBSTACLES:
        path = Path(poly) # Dùng cho point-in-polygon
        for gx in range(size_x_grid):
            for gy in range(size_y_grid):
                world_x = grid_min_x_world + gx * GRID_SIZE
                world_y = grid_min_y_world + gy * GRID_SIZE
                
                # Kiểm tra nếu điểm lưới nằm trong đa giác
                if path.contains_point((world_x, world_y)):
                    grid_map[gx, gy] = 1
                else:
                    # Kiểm tra nếu điểm lưới gần bất kỳ cạnh nào của đa giác (để mở rộng bán kính robot)
                    # Đây là một xấp xỉ, phương pháp chính xác hơn là Minkowski sum
                    num_vertices = len(poly)
                    for i in range(num_vertices):
                        p1 = poly[i]
                        p2 = poly[(i+1) % num_vertices]
                        
                        line_vec = p2 - p1
                        point_vec = np.array([world_x, world_y]) - p1

                        # Độ dài bình phương của đoạn thẳng
                        line_len_sq = np.dot(line_vec, line_vec)
                        if line_len_sq == 0: # Đoạn thẳng suy biến thành điểm
                            if np.linalg.norm(point_vec) < ROBOT_RADIUS:
                                grid_map[gx, gy] = 1
                            continue
                        
                        # Chiếu điểm lên đường thẳng
                        t = np.dot(point_vec, line_vec) / line_len_sq
                        
                        # Điểm gần nhất trên đoạn thẳng
                        if t < 0.0:
                            closest_point = p1
                        elif t > 1.0:
                            closest_point = p2
                        else:
                            closest_point = p1 + t * line_vec
                        
                        distance_to_segment = np.linalg.norm(np.array([world_x, world_y]) - closest_point)
                        
                        if distance_to_segment < ROBOT_RADIUS:
                            grid_map[gx, gy] = 1
                            break # Nếu đã tìm thấy va chạm với 1 cạnh, không cần kiểm tra các cạnh khác

    # Tính toán start_idx và goal_idx
    start_idx = (int((pose[0] - grid_min_x_world) / GRID_SIZE),
                 int((pose[1] - grid_min_y_world) / GRID_SIZE))
    
    goal_idx = (int((goal[0] - grid_min_x_world) / GRID_SIZE),
                int((goal[1] - grid_min_y_world) / GRID_SIZE))

    # Đảm bảo start_idx và goal_idx nằm trong giới hạn lưới
    start_idx = (max(0, min(start_idx[0], size_x_grid - 1)),
                 max(0, min(start_idx[1], size_y_grid - 1)))
    goal_idx = (max(0, min(goal_idx[0], size_x_grid - 1)),
                max(0, min(goal_idx[1], size_y_grid - 1)))
    
    # Kiểm tra và điều chỉnh vị trí start/goal nếu chúng là vật cản
    # (Giữ nguyên logic này từ code cũ của bạn, nhưng đảm bảo nó hoạt động với grid mới)
    if grid_map[start_idx] == 1:
        print(f"Warning: Start at {start_idx} is an obstacle. Trying to find clear spot.")
        found_clear = False
        for dx in range(-EXPAND_SIZE, EXPAND_SIZE + 1):
            for dy in range(-EXPAND_SIZE, EXPAND_SIZE + 1):
                new_sx, new_sy = start_idx[0] + dx, start_idx[1] + dy
                if 0 <= new_sx < size_x_grid and 0 <= new_sy < size_y_grid and grid_map[new_sx, new_sy] == 0:
                    start_idx = (new_sx, new_sy)
                    found_clear = True
                    break
            if found_clear:
                break
        if not found_clear:
            print("Could not find clear spot for start!")

    if grid_map[goal_idx] == 1:
        print(f"Warning: Goal at {goal_idx} is an obstacle. Trying to find clear spot.")
        found_clear = False
        for dx in range(-EXPAND_SIZE_TAR, EXPAND_SIZE_TAR + 1):
            for dy in range(-EXPAND_SIZE_TAR, EXPAND_SIZE_TAR + 1):
                new_gx, new_gy = goal_idx[0] + dx, goal_idx[1] + dy
                if 0 <= new_gx < size_x_grid and 0 <= new_gy < size_y_grid and grid_map[new_gx, new_gy] == 0:
                    goal_idx = (new_gx, new_gy)
                    found_clear = True
                    break
            if found_clear:
                break
        if not found_clear:
            print("Could not find clear spot for goal!")

    return grid_map, start_idx, goal_idx

def openingMap(grid_map):
    rows, cols = grid_map.shape
    expanded_grid = np.copy(grid_map)

    obstacle_rows, obstacle_cols = np.where(grid_map == 1)

    expand_grid_radius = int(ROBOT_RADIUS / GRID_SIZE) + EXPAND_SIZE 
    
    for r, c in zip(obstacle_rows, obstacle_cols):
        for dr in range(-expand_grid_radius, expand_grid_radius + 1):
            for dc in range(-expand_grid_radius, expand_grid_radius + 1):
                if dr*dr + dc*dc <= expand_grid_radius*expand_grid_radius:
                    new_r, new_c = r + dr, c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        expanded_grid[new_r, new_c] = 1 
    return expanded_grid
    
if __name__ == "__main__":
    from robot import Robot
    pose = np.array([3.5, 7.0])

    robots = [Robot(0, np.concatenate([[ 3.5, 7., 5., 0,0,0]]), np.zeros(3)),
              Robot(0, np.concatenate([[11., 6., 5., 0,0,0]]), np.zeros(3))]

    lidar = LidarScanner(range_min=0, range_max=SENSING_RADIUS,
                        angle_min=-math.pi, angle_max=math.pi, resolution=math.pi/90)
    import time
    st = time.time()
    data = lidar.senseObstacle(np.concatenate([pose, [0]]), robots)
    ang, dist = data
    print(time.time()-st)

    # grid_map, start_idx, goal_idx = createGridMap(data, [0,0], [3,5])
    # grid_map = openingMap(grid_map)
    # print(grid_map)
    plt.figure()

    # grid_map[start_idx]=10
    # grid_map[goal_idx] = 20
    # plt.imshow(grid_map, cmap='jet')
    # plt.title("2D LiDAR Grid Map")

    plt.figure()
    # plot robots
    for i in range(len(robots)):
        x, y = robots[i].state[:2]
        a, b = getCircle(x, y, ROBOT_RADIUS)
        plt.plot(a, b, "-b")
    
    # plot obstacle
    for i in range(OBSTACLES.shape[0]):
        x, y, r = OBSTACLES[i,:]
        a, b = getCircle(x, y, r)
        plt.plot(a, b, "-k")

    for poly in POLYGON_OBSTACLES:
        plt.fill(poly[:, 0], poly[:, 1], color='gray', alpha=0.5)
        plt.plot(np.append(poly[:, 0], poly[0,0]), np.append(poly[:, 1], poly[0,1]), 'k-')

    ox = pose[0] + np.cos(ang) * dist
    oy = pose[1] + np.sin(ang) * dist
    plt.scatter(ox, oy)
    plt.axis("equal")
    plt.grid(True)
    plt.show()