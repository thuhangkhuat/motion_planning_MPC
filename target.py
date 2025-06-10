from config import *
from lidar import *
import numpy as np
import matplotlib.pyplot as plt
from planner import AStar
import math
import numpy as np

class Target:
    def __init__(self, initial_state, final_destination):
        self.state = np.array(initial_state, dtype=float) # Initial state [x, y, z]
        self.final_destination = np.array(final_destination, dtype=float)

        self.planned_path = [] #Stores the planned path
        self.path_index = 0 
        self.planner = AStar()
        self.lidar = LidarScanner(range_min=0, range_max=SENSING_RADIUS,
                            angle_min=-math.pi, angle_max=math.pi, resolution=math.pi/45)
        self.velocity = np.zeros(3)
        self.max_speed = TAR_MAX_SPEED

    def generateTrajectory(self,):
        """
        Generates a trajectory from start to goal.
        """
        map_width = int((XLIM[1] - XLIM[0]) / GRID_SIZE)
        map_height = int((YLIM[1] - YLIM[0]) / GRID_SIZE)
        global_grid_map = np.zeros((map_width, map_height))

        def gridCoords(world_pos):
            gx = int((world_pos[0] - XLIM[0]) / GRID_SIZE)
            gy = int((world_pos[1] - YLIM[0]) / GRID_SIZE)
            return gx, gy
        for obs in OBSTACLES:
            # Convert obstacle position to grid coordinates
            for i in range(map_width):
                for j in range(map_height):
                    wx = XLIM[0] + i * GRID_SIZE
                    wy = YLIM[0] + j * GRID_SIZE
                    dist_to_obs = np.hypot(wx - obs[0], wy - obs[1])
                    if dist_to_obs <= obs[2] + GRID_SIZE:
                        global_grid_map[i, j] = 1

        start_grid = gridCoords(self.state)
        goal_grid = gridCoords(self.final_destination)

        # Initialize the A* planner with the global grid map
        self.planner.updatePlanner(global_grid_map, start_grid, goal_grid)
        rx, ry = self.planner.planning()
        rx.reverse()
        ry.reverse()
        for i in range(len(rx)):
            world_x = XLIM[0] + rx[i] * GRID_SIZE
            world_y = YLIM[0] + ry[i] * GRID_SIZE
            self.planned_path.append(np.array([world_x, world_y, self.state[2]]))
        return self.planned_path
    def update(self):
        if self.path_index >= len(self.planned_path):
            self.velocity = np.zeros(3)
            return 
        # Get the next target point in the planned path
        target_waypoint = self.planned_path[self.path_index]
        # Calculate the direction vector to the target waypoint
        direction_vector = target_waypoint - self.state
        distance_to_waypoint = np.linalg.norm(direction_vector[:2])
        if distance_to_waypoint < TAR_EPSILON:
            # If close enough to the waypoint, move to the next one
            self.path_index += 1
            if self.path_index >= len(self.planned_path):
                self.state = self.final_destination
                return
            target_waypoint = self.planned_path[self.path_index]
            direction_vector = target_waypoint - self.state  
        if distance_to_waypoint > 1e-6:
            normalized_direction = direction_vector / np.linalg.norm(direction_vector)
            self.velocity = normalized_direction * self.max_speed
        else: 
            self.velocity = np.zeros(3)
        # Update the target's state
        self.state += self.velocity * TIMESTEP

if __name__ == "__main__":
    def get_circle(x, y, r):
        theta = np.linspace(0, 2 * np.pi, 100)
        a = x + r * np.cos(theta)
        b = y + r * np.sin(theta)
        return a, b
    initial_state = [TAR_STARTS[0], TAR_STARTS[1], TAR_STARTS[2]]
    final_destination = [TAR_GOALS[0], TAR_GOALS[1], TAR_GOALS[2]]
    target = Target(initial_state, final_destination)
    target.generateTrajectory()

   
    plt.ion() 
    fig, ax = plt.subplots(figsize=(12, 6))
    for obs in OBSTACLES:
        x, y, r = obs
        a, b = get_circle(x, y, r)
        ax.fill(a, b, "k", alpha=0.5)
        
    planned_path_np = np.array(target.planned_path)
    ax.plot(planned_path_np[:, 0], planned_path_np[:, 1], 'g--', label='Planned Path (A*)')

    ax.plot(initial_state[0], initial_state[1], 'bo', markersize=10, label='Start')
    ax.plot(final_destination[0], final_destination[1], 'r*', markersize=15, label='Final Destination')
    
    target_path_line, = ax.plot([], [], 'b-', linewidth=2, label='Actual Trajectory')

    current_target_pos, = ax.plot([], [], 'mo', markersize=8, label='Current Target')
    
    ax.set_title("Target Path Simulation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)

    target_path_history = [target.state.copy()] 
    max_steps = 2000
    
    for step in range(max_steps):
        target.update()
        target_path_history.append(target.state.copy())
        
        path_history_np = np.array(target_path_history)
        target_path_line.set_data(path_history_np[:, 0], path_history_np[:, 1])
        current_target_pos.set_data(target.state[0], target.state[1])
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        plt.pause(TIMESTEP)
        
        if np.linalg.norm(target.state - target.final_destination) < 0.1:
            print(f"Target reached destination in {step+1} steps.")
            break
                          
    if step == max_steps - 1:
        print("Simulation ended: Reached max steps.")
        
    plt.ioff() 
    plt.show()