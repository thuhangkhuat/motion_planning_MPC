from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import math

from config import *
from fov import calculate_fov_corners


COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
export = True
if export:
    import cv2
    image_array = []
STANDARD_SIZE = (1280, 720) 

# For drone representation
p1 = np.array([3*ROBOT_RADIUS / 4, 0, 0, 1]).T
p2 = np.array([-3*ROBOT_RADIUS / 4, 0, 0, 1]).T
p3 = np.array([0, 3*ROBOT_RADIUS / 4, 0, 1]).T
p4 = np.array([0,-3*ROBOT_RADIUS / 4, 0, 1]).T

def transformation_matrix(data):
    x, y, z, psi = data[0], data[1], data[2], data[3]
    psi_rotated = psi + np.pi / 4
    c, s = np.cos(psi_rotated), np.sin(psi_rotated)
    return np.array([
        [c, -s, 0, x],
        [s,  c, 0, y],
        [0,  0, 1, z]
    ])

def getCircle(x,y,r):
    theta = np.linspace( 0 , 2 * np.pi , 150 )   
    a = x + r * np.cos( theta )
    b = y + r * np.sin( theta )
    return a, b

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(height_z-5.0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def observerObstacles(pose):
    observed_obstacles = []
    for i in range(OBSTACLES.shape[0]):
        if np.hypot(pose[0]-OBSTACLES[i,0],
                    pose[1]-OBSTACLES[i,1]) < SENSING_RADIUS + OBSTACLES[i,2]:
            observed_obstacles.append(OBSTACLES[i,:])
    return np.array(observed_obstacles)

# path = np.load("path.npy")
# print(path)
with open(FILE_NAME, 'rb') as file:
    data = pickle.load(file)
target_trajectory = data[0]["tar_traj"] 
# path = data["path"]
# predictions = data["predictions"]
# print(predictions.shape)

if SCENARIO == 1:
    size = (8,3.5)
elif SCENARIO == 2:
    size = (12,5.5)
elif SCENARIO == 3:
    size = (12,5.5)
elif SCENARIO == 4:
    size = (12,5.5)
plt.figure(figsize=size)
length = min(data[0]["path"].shape[0], target_trajectory.shape[0])

ax = plt.axes()
for iter in range(length):
    ax.cla()

    # Plot start and goal
    ax.scatter(STARTS[:,0], STARTS[:,1], marker="s", s=50)
    # ax.scatter(TAR_GOALS[0], TAR_GOALS[1], marker="^", s=50, label="Target")

    # Plot obstacles
    kwargs = {'color': 'k', 'linewidth': 1.5, 'linestyle': '-'}
    for j in range(OBSTACLES.shape[0]):
        x, y, r = OBSTACLES[j,:]
        a, b = getCircle(x, y, r)
        ax.plot(a, b, **kwargs)
    ax.plot([], [], label="Obstacles", **kwargs)

    # Plot target trajectory
    ax.plot(target_trajectory[:iter, 0], target_trajectory[:iter, 1], 'r--', label="Target Path")
    ax.plot(target_trajectory[iter, 0], target_trajectory[iter, 1], 'rX', markersize=10, label="Target")

    target_current_pos = target_trajectory[iter]
    circle_x, circle_y = getCircle(target_current_pos[0], target_current_pos[1], VIEWING_RADIUS)
    ax.plot(circle_x, circle_y, linestyle=':', color='green', linewidth=1.5, label=f"Viewing Radius")

    # Plot path
    for i in range(NUM_ROBOT):
        robot_color = COLORS[i % len(COLORS)]
        path = data[i]["path"]
        traj_refs = data[i]["traj_refs"]

        # Plot drone
        robot_current_pos = path[iter, 1:4]
        robot_current_vel = path[iter, 4:6]
        if np.linalg.norm(robot_current_vel) > 1e-5:
            yaw = math.atan2(robot_current_vel[1], robot_current_vel[0])
        else:
            yaw = 0 
        T = transformation_matrix([robot_current_pos[0], robot_current_pos[1], robot_current_pos[2], yaw])
        p1_t = np.matmul(T, p1)
        p2_t = np.matmul(T, p2)
        p3_t = np.matmul(T, p3)
        p4_t = np.matmul(T, p4)

        ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]], 'k-', linewidth=1.5)
        ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]], 'k-', linewidth=1.5)

        ax.scatter([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]], s=50, c='b', marker='o')
        ax.scatter([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]], s=50, c='r', marker='o')


        # Plot path
        plt.plot(path[:iter,1], path[:iter,2], color=robot_color, label="Drone {}".format(i))
        

        # Plot FOV
        robot_current_state = [path[iter, 1:][0],path[iter, 1:][1],3.0]
        fov_corners,_,_ = calculate_fov_corners(robot_current_state, HFOV, VFOV)
        if fov_corners is not None:
            ax.fill(fov_corners[:, 0], fov_corners[:, 1], alpha=0.15, fc=robot_color, ec='none')
            ax.plot(fov_corners[:, 0], fov_corners[:, 1], linestyle='--', color=robot_color, linewidth=1)

        # Plot trajectory reference
        if METHOD == 1:
            plt.plot(traj_refs[iter,:,0], traj_refs[iter,:,1], "k")

    # ax.legend()
    ax.grid(True)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axis("scaled")
    ax.legend()
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    plt.tight_layout()

    plt.gcf().canvas.mpl_connect('key_release_event',
                                    lambda event:
                                    [exit(0) if event.key == 'escape' else None])
    if export:
        file_name = "results/data.png"
        plt.savefig(file_name)
        img = cv2.imread(file_name)
        image_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.pause(0.001)

if export:
    import imageio
    imageio.mimsave(SAVE_GIF, image_array)

plt.show()