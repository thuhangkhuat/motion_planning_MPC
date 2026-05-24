import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from config import *

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None

def is_collision(node_start, node_end, obstacles_rect, obstacles_circ, radius):
    p_start = np.array([node_start.x, node_start.y])
    p_end = np.array([node_end.x, node_end.y])
    for rect in obstacles_rect:
        rect_min = np.array([rect[0] - radius, rect[1] - radius])
        rect_max = np.array([rect[0] + rect[2] + radius, rect[1] + rect[3] + radius])
        num_steps = 20
        for i in range(num_steps + 1):
            p = p_start * (1 - i / num_steps) + p_end * (i / num_steps)
            if rect_min[0] <= p[0] <= rect_max[0] and rect_min[1] <= p[1] <= rect_max[1]:
                return True
    for circ in obstacles_circ:
        center = np.array([circ[0], circ[1]])
        v = p_end - p_start
        a = v.dot(v)
        b = 2 * v.dot(p_start - center)
        c = p_start.dot(p_start) + center.dot(center) - 2 * p_start.dot(center) - (circ[2] + radius)**2
        discriminant = b**2 - 4 * a * c
        if discriminant >= 0:
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / (2 * a)
            t2 = (-b + sqrt_d) / (2 * a)
            if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                return True
    return False


class RRT:
    def __init__(self, waypoints):
        self.s_start = Node(waypoints[0])  # Initial state [x, y]
        self.s_goal = [Node(wp) for wp in waypoints[1:]]
        self.vertex = [self.s_start]
        self.paths = [[] for _ in self.s_goal]

        self.max_iter = TAR_MAX_ITER
        self.step_length = TAR_STEP_LENGTH
        self.goal_sample_rate = TAR_GOAL_SAMPLE_RATE
        self.robot_radius = TAR_RADIUS + SAFETY_MARGIN
        self.x_range = XLIM
        self.y_range = YLIM
        self.obs_rect = RECTANGLE_OBSTACLES
        self.obs_circ = OBSTACLES

    def planning(self):
        for k in range(self.max_iter):
            node_rand = self.generate_random_node()
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not is_collision(node_near, node_new, self.obs_rect, self.obs_circ, self.robot_radius):
                self.vertex.append(node_new)
                for i, goal_node in enumerate(self.s_goal):
                    if len(self.paths[i]) > 0: continue
                    dist, _ = self.get_distance_and_angle(node_new, goal_node)
                    if dist <= self.step_length and not is_collision(node_new, goal_node, self.obs_rect, self.obs_circ, self.robot_radius):
                        goal_node.parent = node_new
                        self.paths[i] = self.extract_path(goal_node)

            if all(len(path) > 0 for path in self.paths):
                return True, self.paths
        
        found_paths = sum(1 for path in self.paths if len(path) > 0)
        return found_paths > 0, self.paths

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        dist = min(self.step_length, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start
        return node_new

    def generate_random_node(self):
        if np.random.random() > self.goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + self.robot_radius, self.x_range[1] - self.robot_radius),
                         np.random.uniform(self.y_range[0] + self.robot_radius, self.y_range[1] - self.robot_radius)))
        return self.s_goal[np.random.randint(len(self.s_goal))]

    def nearest_neighbor(self, node_list, n):
        return min(node_list, key=lambda nd: math.hypot(nd.x - n.x, nd.y - n.y))

    def extract_path(self, node_end):
        path = [[node_end.x, node_end.y]]
        node = node_end
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
    
class SequentialRRTPlanner:
    def __init__(self, waypoints):
        self.waypoints_3d = [np.array(wp) for wp in waypoints]
        self.waypoints_2d = [tuple(wp[:2]) for wp in self.waypoints_3d]
        
        if len(self.waypoints_2d) < 2:
            raise ValueError("Cần ít nhất 2 waypoints (start và goal).")

 
        self.state = None
        self.trajectory = None
        self.final_destination = self.waypoints_3d[-1]
        self.traj_index = 0
        self.obs_rect = RECTANGLE_OBSTACLES
        self.obs_circ = OBSTACLES
        self.clearance_radius = TAR_RADIUS + SAFETY_MARGIN

    def _simplify_path(self, path: list):
        """
        Rút gọn đường đi bằng thuật toán "shortcutting".
        :param path: Đường đi ban đầu (danh sách các điểm [x, y]).
        :return: Đường đi đã được rút gọn.
        """
        if len(path) < 3:
            return path  # Không thể rút gọn đường đi có ít hơn 3 điểm

        simplified_path = [path[0]]
        current_index = 0

        while current_index < len(path) - 1:
            # Tìm điểm xa nhất có thể kết nối trực tiếp
            farthest_reachable_index = current_index + 1
            for lookahead_index in range(len(path) - 1, current_index, -1):
                # Tạo các Node tạm thời để kiểm tra va chạm
                node_start = Node(path[current_index])
                node_end = Node(path[lookahead_index])

                if not is_collision(node_start, node_end, self.obs_rect, self.obs_circ, self.clearance_radius):
                    # Tìm thấy một đường tắt hợp lệ!
                    farthest_reachable_index = lookahead_index
                    break
            
            # Thêm điểm xa nhất tìm được vào đường đi mới
            simplified_path.append(path[farthest_reachable_index])
            # Cập nhật điểm bắt đầu cho lần lặp tiếp theo
            current_index = farthest_reachable_index

        return simplified_path

    def _plan_rrt_path(self):
        """Hàm nội bộ để lập kế hoạch RRT VÀ làm mượt đường đi."""
        full_path_2d = []
        num_segments = len(self.waypoints_2d) - 1
        print(f"[INFO] Planning a path through {num_segments} segments.")

        for i in range(num_segments):
            segment_start = self.waypoints_2d[i]
            segment_goal = self.waypoints_2d[i+1]
            rrt_segment = RRT([segment_start, segment_goal])
            success, paths = rrt_segment.planning()

            if not success or not paths or not paths[0]:
                print(f"[ERROR] Failed to find a path for segment {i+1}. Aborting.")
                return None
            
            # Lấy đường đi thô từ RRT (thứ tự từ goal -> start)
            raw_segment_path = paths[0]
            # Đảo ngược lại để có thứ tự start -> goal
            raw_segment_path.reverse()

            # --- GỌI HÀM LÀM MƯỢT ĐƯỜNG ĐI ---
            print(f"Simplifying path for segment {i+1}... Original nodes: {len(raw_segment_path)}")
            simplified_segment_path = self._simplify_path(raw_segment_path)
            print(f"Simplified path has {len(simplified_segment_path)} nodes.")
            # --- KẾT THÚC BƯỚC LÀM MƯỢT ---

            if i == 0:
                full_path_2d.extend(simplified_segment_path)
            else:
                full_path_2d.extend(simplified_segment_path[1:]) # Bỏ điểm đầu để tránh trùng lặp
        
        print("\n[INFO] RRT path planning and simplification successful.")
        return full_path_2d

    def generateTrajectory(self):
        full_path_2d = self._plan_rrt_path()
        if full_path_2d is None:
            raise RuntimeError("Could not generate target trajectory due to RRT planning failure.")

        traj_gen = TargetTrajectoryGenerator(path=full_path_2d, max_speed=TAR_MAX_SPEED)
        trajectory_2d = list(traj_gen.generate(dt=TIMESTEP))

        z_value = self.waypoints_3d[0][2]
        self.trajectory = [np.array([p[0], p[1], z_value]) for p in trajectory_2d]

        self.state = self.trajectory[0]
        self.traj_index = 0

    def update(self):
    
        self.traj_index += 1
        if self.traj_index < len(self.trajectory):
            self.state = self.trajectory[self.traj_index]
        else:
            self.state = self.trajectory[-1]

class TargetTrajectoryGenerator:
    def __init__(self, path: list, max_speed: float):
        self.path = path
        self.max_speed = max_speed
        self.path_index = 0
        self.current_pos = np.array(self.path[0], dtype=float)

    def generate(self, dt: float):
        yield list(self.current_pos)
        while self.path_index < len(self.path) - 1:
            dist_to_move = self.max_speed * dt
            while dist_to_move > 0 and self.path_index < len(self.path) - 1:
                target_waypoint = np.array(self.path[self.path_index + 1],dtype=float)
                vector_to_target = target_waypoint - self.current_pos
                dist_to_waypoint = np.linalg.norm(vector_to_target)
                if dist_to_waypoint == 0:
                    self.path_index += 1
                    continue
                if dist_to_move >= dist_to_waypoint:
                    self.current_pos = target_waypoint.astype(float)
                    dist_to_move -= dist_to_waypoint
                    self.path_index += 1
                else:
                    direction = vector_to_target / dist_to_waypoint
                    self.current_pos += direction * dist_to_move
                    dist_to_move = 0
            yield list(self.current_pos)

def main():

    fig, ax = plt.subplots(figsize=(10, 10))
    # waypoints = [tuple(wp) for wp in TAR_WAYPOINTS]
    
    for rect in RECTANGLE_OBSTACLES:
        ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], facecolor='gray', edgecolor='black'))
    for circ in OBSTACLES:
        ax.add_patch(Circle((circ[0], circ[1]), circ[2], facecolor='gray', edgecolor='black'))

    ax.set_title("Sequential RRT Planning")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()