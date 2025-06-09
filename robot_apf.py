import numpy as np
import casadi as ca

from lidar import LidarScanner
from config import *

class Robot:
    def __init__(self, index, state:np.array, goal:np.array, control=np.zeros(3)):
        # Robot state and control
        self.time_stamp = 0.0
        
        self.index = index
        self.state = state
        self.control = control
        self.goal = goal

        self.n_state = 6
        self.n_control = 3

        self.lidar = LidarScanner(range_min=0, range_max=SENSING_RADIUS,
                                  angle_min=-np.pi, angle_max=np.pi, resolution=np.pi/45)
        
        # Store robot path
        self.path = []
        self.traj_refs = []
        # self.path = [np.concatenate([[self.time_stamp], self.state, self.control])]

        self.states_prediction = np.ones((HORIZON_LENGTH+1, self.n_state))*self.state
        self.controls_prediction = np.zeros((HORIZON_LENGTH, self.n_control))

    def updateState(self, control:np.array, dt:float):
        """
        Computes the states of robot after applying control signals
        """
        
        # Update
        position = self.state[:3]
        velocity = self.state[3:6]

        next_position = position + velocity*dt
        next_velocity = velocity + (control-D_FRAC*velocity)*dt
        if np.linalg.norm(next_velocity) > VMAX:
            next_velocity = next_velocity/np.linalg.norm(next_velocity)*VMAX

        self.state = np.concatenate([next_position, next_velocity])
        self.control = control
        self.time_stamp = self.time_stamp + dt

        # Store
        self.path.append(np.concatenate([[self.time_stamp], self.state, self.control]))
        self.traj_refs.append(self.goal)

        # Shift predictive values
        self.states_prediction[:-1,:] = self.states_prediction[1:,:]
        self.controls_prediction[:-1,:] = self.controls_prediction[1:,:]

    def computeControlSignal(self, robots):
        """
        Computes control velocity of the copter
        """
        scan_data = self.lidar.senseObstacle(np.concatenate([self.state[:2], [0]]), robots)
        v_tar = 1.0*self.behaviorTarget()
        v_col = 1.5*self.behaviorCollision(scan_data)
        v_ran = 1e-2*self.behaviorRandom()
        v = v_tar + v_col + v_ran
        
        control = v - self.state[3:]
        if np.linalg.norm(control) > UMAX:
            control = control/np.linalg.norm(control)*UMAX
        self.updateState(control, TIMESTEP)

    def behaviorTarget(self):
        dist = np.linalg.norm(self.goal - self.state[:3])
        if dist < 1.0:
            return (self.goal - self.state[:3])
        else:
            return (self.goal - self.state[:3])/dist
    
    def behaviorCollision(self, scan_data):
        v_col = 0
        count = 0
        ang, dist = scan_data
        if dist.shape[0] != 0:
            num = len(dist)
            for j in range(num):
                obs_x = dist[j] * np.cos(ang[j]) + self.state[0]
                obs_y = dist[j] * np.sin(ang[j]) + self.state[1]
                obs = np.array([obs_x, obs_y])
                
                obs_rel = self.state[:2] - obs[:2]
                dis = np.linalg.norm(obs_rel) - ROBOT_RADIUS
                if dis > 1.0:
                    continue
                count +=1
                dir = np.concatenate([obs_rel,[0]])/dis
                v_col += (1.0-dis)/(1.0-ROBOT_RADIUS)*dir
        if count == 0:
            count = 1
        return v_col/count
    
    def behaviorRandom(self):
        return np.random.rand(self.n_control)