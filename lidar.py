import math
import numpy as np
import matplotlib.pyplot as plt

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

    def isCollision(self, pose, obstacles):    
        for (cx, cy, cr) in obstacles:
            ex = cx - pose[0]; ey = cy - pose[1]
            if ex**2 + ey**2 - cr**2 <= 0:
                return True
        return False

    def senseObstacle(self, pose, robots):
        obstacles = []
        # Sense obstacles
        for i in range(OBSTACLES.shape[0]):
            if np.hypot(pose[0]-OBSTACLES[i,0],
                        pose[1]-OBSTACLES[i,1]) < SENSING_RADIUS + OBSTACLES[i,2]:
                obstacles.append(OBSTACLES[i,:])
        
        # Sense robots
        # for i in range(len(robots)):
        #     distance = np.hypot(pose[0]-robots[i].state[0], pose[1]-robots[i].state[1])
        #     if distance < SENSING_RADIUS + ROBOT_RADIUS and distance > ROBOT_RADIUS:
        #         obstacles.append(np.array([robots[i].state[0],
        #                                    robots[i].state[1],
        #                                    ROBOT_RADIUS]))

        data = []
        x, y, _ = pose
        for angle in np.linspace(self.angle_min + pose[2], self.angle_max + pose[2],
                                self.angle_num, True):
            sense = False
            x1 = x + self.range_min*math.cos(angle)
            y1 = y + self.range_min*math.sin(angle)

            x2 = x + self.range_max*math.cos(angle)
            y2 = y + self.range_max*math.sin(angle)

            for i in range(self.range_num+1):
                u = i/self.range_num
                x3 = x2*u + x1*(1 - u)
                y3 = y2*u + y1*(1 - u)
                if self.isCollision([x3, y3], obstacles):
                    dis = self.distance(pose, [x3, y3])
                    data.append(dis+np.random.rand()*self.noise)
                    sense = True
                    break
            if not sense:
                data.append(self.range_max)
        
        angle = np.linspace(self.angle_min, self.angle_max, self.angle_num)
        data = np.array(data)

        idx = np.where(data<self.range_max)
        angle = angle[idx]
        data = data[idx]
        return angle, data
    
    def getObstaclePoints(self, pose, obstacles):
        scan_angles, scan_ranges = self.senseObstacle(pose, obstacles)
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
    from robot import Robot
    pose = np.array([-0.5, 0.0])

    robots = [Robot(0, np.concatenate([[ .5, 0., 5., 0,0,0]]), np.zeros(3)),
              Robot(0, np.concatenate([[11., 6., 5., 0,0,0]]), np.zeros(3))]

    lidar = LidarScanner(range_min=0, range_max=SENSING_RADIUS,
                        angle_min=-math.pi, angle_max=math.pi, resolution=math.pi/45)
    import time
    st = time.time()
    data = lidar.senseObstacle(np.concatenate([pose, [0]]), robots)
    ang, dist = data
    print(time.time()-st)

    grid_map, start_idx, goal_idx = createGridMap(data, [0,0], [3,5])
    grid_map = openingMap(grid_map)
    # print(grid_map)
    plt.figure()

    grid_map[start_idx]=10
    grid_map[goal_idx] = 20
    plt.imshow(grid_map, cmap='jet')
    plt.title("2D LiDAR Grid Map")

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

    ox = pose[0] + np.cos(ang) * dist
    oy = pose[1] + np.sin(ang) * dist
    plt.scatter(ox, oy)
    plt.axis("equal")
    plt.grid(True)
    plt.show()