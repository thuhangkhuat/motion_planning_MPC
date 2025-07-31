import numpy as np
import casadi as ca

import pydecomp as pdc

from lidar import LidarScanner
from jps import JumpPointSearch
from utils import HeuristicType

from config import *

import matplotlib.pyplot as plt

class Robot:
    def __init__(self, index, state:np.array, goal:np.array, control=np.zeros(3)):
        # Robot state and control
        self.time_stamp = 0.0
        
        self.index = index
        self.state = state
        self.control = control
        self.goal = goal

        self.planner_update_counter = 0
        self.PLANNER_UPDATE_RATE = 1

        self.n_state = 6
        self.n_control = 3
        self.planner = JumpPointSearch()

        self.lidar = LidarScanner(range_min=0, range_max=SENSING_RADIUS,
                                  angle_min=-np.pi, angle_max=np.pi, resolution=np.pi/45)
        
        #Store the corridor
        self.corridors = []
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

        self.state = np.concatenate([next_position, next_velocity])
        self.control = control
        self.time_stamp = self.time_stamp + dt

        # Store
        self.path.append(np.concatenate([[self.time_stamp], self.state, self.control]))
        self.traj_refs.append(self.traj_ref)

        # Shift predictive values
        self.states_prediction[:-1,:] = self.states_prediction[1:,:]
        self.controls_prediction[:-1,:] = self.controls_prediction[1:,:]

    def computeControlSignal(self, robots):
        """
        Computes control velocity of the copter
        """
        scan_data = self.lidar.senseObstacle(np.concatenate([self.state[:2], [0]]), robots)
        self.traj_ref = self.getOrientedGoalTrajectory(scan_data, self.goal)
        obstacle_points = self.lidar.getObstaclePoints(np.concatenate([self.state[:2], [0]]), OBSTACLES)
        target_pos = self.goal[:3].reshape(1, 3) 
        list_A, list_b = self.generateSafeCorridor(self.traj_ref, obstacle_points)
        neighbor_robots = self.getNeighbors(robots)
        
        opti = ca.Opti()
        # control variables, linear velocity and angular velocity
        opt_states = opti.variable(HORIZON_LENGTH+1, self.n_state)
        opt_controls = opti.variable(HORIZON_LENGTH, self.n_control)
        slack_cbf = opti.variable(HORIZON_LENGTH, 1)

        f = lambda x_, u_: ca.horzcat(*[
            x_[3:],
            u_
        ])
        # f = lambda x_, u_: ca.horzcat(x_[3:], u_ - D_FRAC * x_[3:])


        opti.subject_to(opt_states[0, :] == np.array([self.state]))
        for i in range(HORIZON_LENGTH):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :])*TIMESTEP
            opti.subject_to(opt_states[i+1, :] == x_next)

        # add constraints with conver polygon -> liner constraints
        if list_A:
            active_A, active_b = None, None
            for A, b in zip(list_A, list_b):
                if np.all(A @ self.state[:2] - b.flatten() <= 1e-5):
                    active_A = A
                    active_b = b
                    break
            self.corridors.append({'A': active_A, 'b': active_b})
            if active_A is not None:
                for i in range(HORIZON_LENGTH + 1):
                    opti.subject_to(ca.mtimes(active_A, opt_states[i, :2].T) <= active_b)

        # add constraints to obstacle
        ang, dist = scan_data
        if dist.shape[0] != 0:
            min_idx = np.argmin(dist)
            obs_x = dist[min_idx] * np.cos(ang[min_idx]) + self.state[0]
            obs_y = dist[min_idx] * np.sin(ang[min_idx]) + self.state[1]
            for i in range(HORIZON_LENGTH+1):
                # MPC constraint
                temp_constraints_ = ca.sqrt((opt_states[i,0]-obs_x)**2 + \
                                            (opt_states[i,1]-obs_y)**2) - ROBOT_RADIUS
                opti.subject_to(temp_constraints_ > 0.0)

        # add constrain to neighbors robot
        for i in range(HORIZON_LENGTH):
            for other_robot in neighbor_robots:
                if self.index >= other_robot.index: 
                    continue
                other_pos = ca.reshape(ca.DM(other_robot.states_prediction[i, :2]), 1, 2)
                # MPC constraint
                dist_sq = ca.sumsqr(opt_states[i, :2] - other_pos)
                opti.subject_to(dist_sq >= (2*ROBOT_RADIUS)**2)

        # add constraints CBF and formation
        for i in range(HORIZON_LENGTH):
            current_state = opt_states[i, :]
            next_state = opt_states[i+1, :]
            current_pos = current_state[:3]
            next_pos = next_state[:3]
            h_k = VIEWING_RADIUS**2 - ca.sumsqr(current_pos - target_pos)
            h_k_plus = VIEWING_RADIUS**2 - ca.sumsqr(next_pos - target_pos)
            
            # opti.subject_to(h_k_plus - (1 - DT_CBF_GAMMA) * h_k >= 0)
            opti.subject_to(h_k_plus - (1 - DT_CBF_GAMMA) * h_k >= -slack_cbf[i])

        # velocity and control constraints
        for i in range(HORIZON_LENGTH):
            vel = opt_states[i+1,3:]
            vel_sq = VMAX**2 - ca.mtimes([vel, vel.T])
            opti.subject_to(vel_sq >= 0)

            con = opt_controls[i,:]
            con_sq = UMAX**2 - ca.mtimes([con, con.T])
            opti.subject_to(con_sq >= 0)
        for i in range(HORIZON_LENGTH):
            vel = opt_states[i+1, 3:]
            opti.subject_to(ca.sumsqr(vel) <= VMAX**2)
            con = opt_controls[i, :]
            opti.subject_to(ca.sumsqr(con) <= UMAX**2)
        
        opts_setting = {'ipopt.max_iter': 2000,   #1e5
                        'ipopt.print_level': 0,
                        'ipopt.tol': 1e-4,  #1e-6
                        'ipopt.acceptable_tol': 1e-2,  #1e-6
                        'print_time': 0}
        opti.solver('ipopt', opts_setting)

        # cost function
        obj = self.costFunction(opt_states, opt_controls, scan_data,slack_cbf, neighbor_robots)
        opti.minimize(obj)

        # provide the initial guess of the optimization targets
        opti.set_initial(opt_states, self.states_prediction)
        opti.set_initial(opt_controls, self.controls_prediction)

        # solve the problem
        sol = opti.solve()
        
        ## obtain the control input
        self.controls_prediction = sol.value(opt_controls)
        self.states_prediction = sol.value(opt_states)
        
        # return self.controls_prediction[0,:]
        control = self.controls_prediction[0,:]
        self.updateState(control, TIMESTEP)

    def getOrientedGoalTrajectory(self, data, goal):
        position = self.state[:3]
        print(f"Current position: {position}, Goal: {goal}")
        grid_map, start_idx, goal_idx = self.createGridMap(data, position, goal)
        print(start_idx, goal_idx)
        grid_map = self.openingMap(grid_map)
        origin_x = grid_map.shape[0] // 2
        origin_y = grid_map.shape[1] // 2
        path_indices = self.planner.plan(grid_map.T, start_idx, goal_idx)
        traj_ref = []
            
        for p_idx in path_indices:
            p_grid_x, p_grid_y = p_idx
            delta_world_x = (p_grid_x - origin_x) * GRID_SIZE

            delta_world_y = (p_grid_y - origin_y) * GRID_SIZE
            p_world_x = position[0] + delta_world_x
            p_world_y = position[1] + delta_world_y
            
            traj_ref.append([p_world_x, p_world_y])
        # traj_ref.append(goal[:2])  # Append the goal position at the end
        traj_ref = np.array(traj_ref)
        return np.array(traj_ref)

    def costFunction(self, opt_states, opt_controls, scan_data, slack_vars, neighbors):
        c_u = self.costControl(opt_controls)
        c_tra = self.costTracking(opt_states)
        # c_col = self.costCollision(opt_states, scan_data)
        c_col = 0
        c_form = self.costFormation(opt_states, neighbors)
        c_slack = self.costSlack(slack_vars) 
        total = c_tra + c_u + c_col + c_slack + c_form

        return total
    
    def costSlack(self, slack_vars):
        return W_slack * ca.sumsqr(slack_vars)

    def costControl(self, u):
        cost_u = 0
        for i in range(HORIZON_LENGTH):
            control = u[i,:]
            # cost_u = ca.mtimes(control, control.T)
            cost_u += ca.sumsqr(control)
        return W_u*cost_u

    def costTracking(self, traj):
        cost_tra = 0
        dist_goal = ca.sumsqr(traj[-1, :2] - self.goal[:2].reshape(1, 2))

        cost_tra += (dist_goal - (VIEWING_RADIUS -0.5)**2)**2
        return W_tra*cost_tra
    
    def costCollision(self, traj, scan_data):
        cost_col = 0
        ang, dist = scan_data
        if dist.shape[0] != 0:
            min_idx = np.argmin(dist)
            obs_x = dist[min_idx] * np.cos(ang[min_idx]) + self.state[0]
            obs_y = dist[min_idx] * np.sin(ang[min_idx]) + self.state[1]
            for i in range(HORIZON_LENGTH):
                # obs_rel = traj[i,:2].T - np.array([obs_x, obs_y])
                # cost_col += 1./(1+ca.exp(4*(ca.mtimes(obs_rel.T, obs_rel) - ROBOT_RADIUS)))
                # cost_col -= ca.log(ca.sumsqr(obs_rel) - ROBOT_RADIUS**2)
                dist_sq = ca.sumsqr(traj[i,:2] - ca.DM([obs_x, obs_y]).T)
                margin = dist_sq - ROBOT_RADIUS**2
                cost_col += 1 / (margin + 1e-4)
        return W_col*cost_col
    
    def costFormation(self, traj, neighbors):
        cost_dist = 0 
        cost_spread = 0
        if not neighbors:
            cost_dist = 0
            cost_spread = 0
        current_predicted_pos = self.states_prediction[:, :2]
        min_dist_sq_avg = float('inf')
        nearest_neighbor = None
        other_neighbors = []
        for other_robot in neighbors:
            if self.index >= other_robot.index:
                continue
            other_predicted_pos = other_robot.states_prediction[:, :2]
            avg_dist_sq = np.mean(np.sum((current_predicted_pos - other_predicted_pos)**2, axis=1))
            if avg_dist_sq < min_dist_sq_avg:
                if nearest_neighbor is not None:
                    other_neighbors.append(nearest_neighbor)
                min_dist_sq_avg = avg_dist_sq
                nearest_neighbor = other_robot
            else:
                other_neighbors.append(other_robot)
        for i in range(HORIZON_LENGTH):
            current_pos = traj[i, :2]
            if nearest_neighbor is not None:
                other_pos = ca.reshape(ca.DM(nearest_neighbor.states_prediction[i, :2]), 1, 2)
                dist_sq = ca.sumsqr(current_pos - other_pos)
                cost_dist += (dist_sq - DESIRED_SEPARATION**2)**2

            for other_robot in other_neighbors:
                other_pos = ca.reshape(ca.DM(other_robot.states_prediction[i, :2]),1,2)
                dist_sq_other = ca.sumsqr(current_pos - other_pos)
                cost_spread -= dist_sq_other
        return W_form_dist*cost_dist + W_form_spread*cost_spread

    def predictTrajectory(self, state, controls):
        """
        Computes the states of the system after applying a sequence of control signals u on
        initial state x0
        """
        trajectory = []
        for i in range(HORIZON_LENGTH):
            # Update
            position = state[:3]
            velocity = state[3:]
            control = controls[self.n_control*i:self.n_control*(i+1)]
            next_position = position + velocity*TIMESTEP
            next_velocity = velocity + (control-D_FRAC*velocity)*TIMESTEP 
            state = np.concatenate([next_position, next_velocity])
            trajectory.append(state)
        return np.array(trajectory)
    
    def getNeighbors(self,robots):
        neighbors = []
        current_pos = self.state[:3]
        for other_robot in robots:
            if self.index == other_robot.index:
                continue
            other_current_pos = other_robot.state[:3]
            distance = np.linalg.norm(current_pos - other_current_pos)
            # print(f"Distance to robot {other_robot.index}: {distance}")
            if distance < SENSING_NEIGHBOR:
                neighbors.append(other_robot)
        return neighbors



    def generateSafeCorridor(self,path_ref, obstacle_points):
        """
        Create convex polygon using pydecomp
        """
        # print(obstacle_points)
        if obstacle_points.shape[0] < 1: 
            return [], []

        box = np.array([[VIEWING_RADIUS, VIEWING_RADIUS]])

        try:
            list_A, list_b = pdc.convex_decomposition_2D(obstacle_points, path_ref, box)
            return list_A, list_b
        except Exception as e:
            print(f"Error in generating safe corridor: {e}")
            return [], []
    
    @staticmethod
    def createGridMap(data, pose, goal):
        size_x = int(2*max(SENSING_RADIUS, abs(goal[0]-pose[0]))/GRID_SIZE)+1
        size_y = int(2*max(SENSING_RADIUS, abs(goal[1]-pose[1]))/GRID_SIZE)+1

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
                

        # Start and goal indexes
        start_idx = (origin_x, origin_y)
        goal_idx = (int(origin_x + (goal[0]-pose[0]) / GRID_SIZE),
                    int(origin_y + (goal[1]-pose[1]) / GRID_SIZE))
        return grid_map, start_idx, goal_idx

    @staticmethod
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
    
