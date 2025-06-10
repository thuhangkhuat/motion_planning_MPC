import numpy as np
import casadi as ca

from lidar import LidarScanner
from planner import AStar

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

        self.lidar = LidarScanner(range_min=0, range_max=SENSING_RADIUS,
                                  angle_min=-np.pi, angle_max=np.pi, resolution=np.pi/45)
        self.planner = AStar()
        
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
        # if self.planner_update_counter % self.PLANNER_UPDATE_RATE == 0:
        #     self.traj_ref = self.getOrientedGoalTrajectory(scan_data, self.goal)
        # self.planner_update_counter += 1

        target_pos = self.goal[:3].reshape(1, 3) 
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

        # add constraints CBF and formation
        for i in range(HORIZON_LENGTH):
            current_state = opt_states[i, :]
            current_pos = current_state[:3]
            current_vel = current_state[3:]
            current_control = opt_controls[i, :]
            # Second-Order CBF
            h_target = VIEWING_RADIUS**2 - ca.sumsqr(current_pos - target_pos)
            h_dot_target = -2 * ca.mtimes((current_pos - target_pos).T, current_vel)
            
            Lf2_h_target = -2 * ca.sumsqr(current_vel)
            LgLf_h_target = -2 * (current_pos - target_pos)
            # h_ddot + 2*gamma*h_dot + gamma^2*h >= 0
            opti.subject_to(Lf2_h_target + ca.mtimes(LgLf_h_target, current_control.T) + \
                            2 * CBF_GAMMA * h_dot_target + CBF_GAMMA**2 * h_target >= slack_cbf[i])
            opti.subject_to(slack_cbf[i] >= 0)


        # velocity and control constraints
        for i in range(HORIZON_LENGTH):
            vel = opt_states[i+1,3:]
            vel_sq = VMAX**2 - ca.mtimes([vel, vel.T])
            opti.subject_to(vel_sq >= 0)

            con = opt_controls[i,:]
            con_sq = UMAX**2 - ca.mtimes([con, con.T])
            opti.subject_to(con_sq >= 0)

        opts_setting = {'ipopt.max_iter': 200,   #1e5
                        'ipopt.print_level': 0,
                        'ipopt.tol': 1e-4,  #1e-6
                        'ipopt.acceptable_tol': 1e-3,  #1e-6
                        'print_time': 0}
        opti.solver('ipopt', opts_setting)

        # cost function
        obj = self.costFunction(opt_states, opt_controls, self.traj_ref, scan_data,slack_cbf, neighbor_robots)
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
        grid_map, start_idx, goal_idx = self.createGridMap(data, position, goal)
        grid_map = self.openingMap(grid_map)
        self.planner.updatePlanner(grid_map, start_idx, goal_idx)
        rx, ry = self.planner.planning()
        rx = rx[::-1][1:]; ry = ry[::-1][1:]

        # Trajectory
        path = []

        for i in range(HORIZON_LENGTH):
            if i < len(rx):
                x = rx[i]; y = ry[i]
                path.append(np.array([(x - grid_map.shape[0]//2) * GRID_SIZE,
                                      (y - grid_map.shape[1]//2) * GRID_SIZE,
                                      0.0]) + self.state[:3])
            else:
                path.append(goal)
        return np.array(path)

    def costFunction(self, opt_states, opt_controls, traj_ref, scan_data, slack_vars, neighbors):
        c_u = self.costControl(opt_controls)
        c_tra = self.costTracking(opt_states, traj_ref)
        c_col = self.costCollision(opt_states, scan_data)
        c_form = self.costFormation(opt_states, neighbors)
        c_slack = self.costSlack(slack_vars) 
        total = c_tra + c_u + c_col + c_slack + W_form_dist * c_form['dist'] + W_form_struct * c_form['struct']

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

    def costTracking(self, traj, traj_ref):
        cost_tra = 0
        for i in range(HORIZON_LENGTH):
            pos_rel = traj[i,:3].T - traj_ref[i,:3]
            cost_tra += ca.mtimes(pos_rel.T, pos_rel)
        return W_tra*cost_tra
    
    def costCollision(self, traj, scan_data):
        cost_col = 0
        ang, dist = scan_data
        if dist.shape[0] != 0:
            min_idx = np.argmin(dist)
            obs_x = dist[min_idx] * np.cos(ang[min_idx]) + self.state[0]
            obs_y = dist[min_idx] * np.sin(ang[min_idx]) + self.state[1]
            for i in range(HORIZON_LENGTH):
                obs_rel = traj[i,:2].T - np.array([obs_x, obs_y])
                # cost_col += 1./(1+ca.exp(4*(ca.mtimes(obs_rel.T, obs_rel) - ROBOT_RADIUS)))
                cost_col -= ca.log(ca.sumsqr(obs_rel) - ROBOT_RADIUS**2)
        return W_col*cost_col
    
    def costFormation(self, traj, neighbors):
        cost_dist = 0 
        cost_struct = 0
        if not neighbors:
            return {'dist': 0.0, 'struct': 0.0}
        for i in range(HORIZON_LENGTH):
            current_pos = traj[i, :3]
            for other_robot in neighbors:
                if self.index >= other_robot.index: 
                    continue
                other_pos = other_robot.states_prediction[i, :3]
                dist_sq = ca.sumsqr(current_pos - other_pos)
                cost_dist += ca.exp(-5 * (dist_sq - MIN_SEPARATION**2))
                cost_dist += ca.exp(5 * (dist_sq - MAX_SEPARATION**2))
        for i in range(HORIZON_LENGTH):
            pos_i = traj[i, :3]
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    robot_j = neighbors[j]
                    robot_k = neighbors[k]
                    pos_j = robot_j.states_prediction[i, :3]
                    pos_k = robot_k.states_prediction[i, :3]
                    vec_ij = pos_j - pos_i
                    vec_ik = pos_k - pos_i
                    area_sq = 0.25 * (vec_ij[0]*vec_ik[1] - vec_ij[1]*vec_ik[0])**2
                    epsilon = 1e-6
                    cost_struct -= ca.log(area_sq + epsilon)

        return {'dist': cost_dist, 'struct': cost_struct}

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
            if distance < SENSING_NEIGHBOR:
                neighbors.append(other_robot)
        return neighbors
    
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
    
    
