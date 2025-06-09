from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pickle

from config import *

def getCircle(x,y,r):
    theta = np.linspace( 0 , 2 * np.pi , 50 )   
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

with open('data.txt', 'rb') as file:
    data = pickle.load(file)

path = data["path"]

## Plot motion path in 2D
plt.figure(figsize=(11,3))
# ax = plt.axes(projection="3d")
ax = plt.axes()

# Plot obstacles
for j in range(OBSTACLES.shape[0]):
    x, y, r = OBSTACLES[j,:]
    a, b = getCircle(x, y, r)
    ax.plot(a, b, '-k')
    a, b = getCircle(x, y, r+ROBOT_RADIUS)
    ax.plot(a, b, '--k', linewidth="0.5")

# Plot path
points = path[::2,1:3].reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
speed = np.linalg.norm(path[:,4:7], axis=1)

norm = plt.Normalize(speed.min(), speed.max())
lc = LineCollection(segments, cmap='rainbow', norm=norm)
# Set the values used for colormapping
lc.set_array(speed)
lc.set_linewidth(2)
line = ax.add_collection(lc)
bar = plt.colorbar(line)
bar.set_label('Speed [m/s]', rotation=270)
bar.ax.get_yaxis().labelpad = 15

# Plot start and goal
ax.scatter(START[0], START[1], marker="s", s=50, label="Start")
ax.scatter(GOAL[0], GOAL[1], marker="^", s=50, label="Goal")

# ax.legend()
ax.grid(True)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.axis("scaled")
ax.legend()
ax.set_xlim([0, 35])
ax.set_ylim([0, 10])
plt.tight_layout()

plt.savefig("results/motion_path_our_top.pdf", format="pdf")

## Plot motion path in 3D
fig = plt.figure(figsize=(10,9))
ax = plt.axes(projection="3d")

for i in range(OBSTACLES.shape[0]):
    x, y, r = OBSTACLES[i,:]
    Xc,Yc,Zc = data_for_cylinder_along_z(x, y, r, 8.0)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color="r")

# Plot path
ax.plot(path[:,1], path[:,2], path[:,3])

# Plot start and goal
ax.scatter(START[0], START[1], START[2], marker="s", label="Start")
ax.scatter(GOAL[0], GOAL[1], GOAL[2], marker="^", label="Goal")

ax.azim = -100
ax.elev = 25
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
# ax.axis("equal")
# ax.legend()
ax.set_xlim([0, 35])
ax.set_ylim([0, 10])
ax.set_zlim([3, 8])
ax.set_box_aspect(aspect=(3.5, 1.0, .5), zoom=1.0)
plt.tight_layout()

## Plot speed
plt.figure(figsize=(5,3))
plt.plot(path[:,0], np.linalg.norm(path[:,4:7], axis=1), "-", label="Speed profile")
plt.plot([path[0,0], path[-1,0]], [VREF, VREF], "--k", label=r'$v_{ref}$')
plt.legend()
plt.ylabel("Speed [m/s]")
plt.xlabel("Time [s]")
plt.xlim([path[0,0], path[-1,0]])
plt.ylim([0, 1.2])
plt.tight_layout()

plt.show()