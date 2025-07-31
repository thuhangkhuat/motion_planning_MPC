import math
import numpy as np
import matplotlib.pyplot as plt
from config import *
from utils import *
from lidar import *
    
class RRT:
    def __init__(self, x_start:Node, x_goals:Node,obstacle_points:np.ndarray):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goals)
        self.obstacle_points = obstacle_points

        self.vertex = [self.s_start]
        self.path = []

    def planning(self):
        for _ in range(MAX_ITER):
            node_rand = self.generate_random_node()
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not is_collision(node_near, node_new, self.obstacle_points, ROBOT_RADIUS):
                self.vertex.append(node_new)

                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)
                if dist <= STEP_LENGTH and not is_collision(node_new, self.s_goal,self.obstacle_points, ROBOT_RADIUS):
                    index = self.search_goal_parent()
                    self.path = self.extract_path(self.vertex[index])
                    return True, self.path
        return False, []

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(STEP_LENGTH, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= STEP_LENGTH]
        if len(node_index) > 0:
            cost_list = []
            valid_indices = []
            for i in node_index:
                if not is_collision(self.vertex[i], self.s_goal, self.obstacle_points, ROBOT_RADIUS):
                    cost_list.append(dist_list[i] + self.cost(self.vertex[i]))
                    valid_indices.append(i)
            
            if not cost_list:
                return len(self.vertex) - 1

            return valid_indices[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)

        return self.cost(node_start) + dist

    def generate_random_node(self):
        if np.random.random() > GOAL_SAMPLE_RATE:
            return Node((np.random.uniform(XLIM[0] + ROBOT_RADIUS, XLIM[1] - ROBOT_RADIUS),
                         np.random.uniform(YLIM[0] + ROBOT_RADIUS, YLIM[1] - ROBOT_RADIUS)))

        return self.s_goal

    def nearest_neighbor(self, node_list:Node, n):
        
        dis_node = [math.hypot(nd.x - n.x, nd.y - n.y) for nd in node_list]

        return node_list[int(np.argmin(dis_node))]

    @staticmethod
    def cost(node_p:Node):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path
    
    def remove_residual_node(path, start, goal, obstacle_points, ROBOT_RADIUS):
        new_path = [start]
        count = 500
        while not np.array_equal(new_path[-1], goal) and count > 0:
            for i in range(len(path)):
                if not is_collision(Node(new_path[-1]), Node(path[i]), obstacle_points, ROBOT_RADIUS):
                    new_path.append(path[i])
                    break
            count -= 1
        return count, new_path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
if __name__ == "__main__":
    from robot import Robot
    pose = np.array([6.2,5.0])
    goal = np.array([12,5.5])
    robots = [Robot(0, np.concatenate([[-2.5, 0., 5., 0,0,0]]), np.zeros(3)),
              Robot(0, np.concatenate([[11., 6., 5., 0,0,0]]), np.zeros(3))]

    lidar = LidarScanner(range_min=0, range_max=SENSING_RADIUS,
                        angle_min=-math.pi, angle_max=math.pi, resolution=math.pi/90)
    import time
    st = time.time()
    data = lidar.senseObstacle(np.concatenate([pose, [0]]), robots)
    print(data)
    obstacle_points = lidar.getObstaclePoints(data, pose)
    rrt  = RRT(pose, goal, obstacle_points)
    success, raw_path = rrt.planning()
    count, smoothed_path = RRT.remove_residual_node(raw_path, pose, goal, obstacle_points, ROBOT_RADIUS)
    print(time.time()-st)
    if success:
        print("Path found:", raw_path)
    else:
        print("No path found")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("RRT Path Planning Results")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)

    # 2. Vẽ các điểm vật cản từ Lidar (obstacle_points)
    if obstacle_points.size > 0:
        ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], s=15, c='red', 
                   label='Lidar Scan Points')

    # 3. Vẽ đường đi thô (raw_path)
    if raw_path:
        path_x = [p[0] for p in raw_path]
        path_y = [p[1] for p in raw_path]
        ax.plot(path_x, path_y, 'r.--', label='Raw RRT Path')

    # 4. Vẽ đường đi đã làm mượt (smoothed_path / residual path)
    if smoothed_path:
        path_x = [p[0] for p in smoothed_path]
        path_y = [p[1] for p in smoothed_path]
        ax.plot(path_x, path_y, 'g-', linewidth=2, label='Smoothed Path')
    
    # 5. Vẽ điểm bắt đầu và đích
    ax.plot(pose[0], pose[1], 'go', markersize=12, label='Start')
    ax.plot(goal[0], goal[1], 'm*', markersize=15, label='Goal')

    # 6. Hiển thị chú thích và đồ thị
    ax.legend()
    plt.show()
    