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
        
        opti = ca.Opti()
        # control variables, linear velocity and angular velocity
        opt_states = opti.variable(HORIZON_LENGTH+1, self.n_state)
        opt_controls = opti.variable(HORIZON_LENGTH, self.n_control)

        f = lambda x_, u_: ca.horzcat(*[
            x_[3:],
            u_
        ])

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
        
        # velocity and control constraints
        for i in range(HORIZON_LENGTH):
            vel = opt_states[i+1,3:]
            vel_sq = VMAX**2 - ca.mtimes([vel, vel.T])
            opti.subject_to(vel_sq >= 0)

            con = opt_controls[i,:]
            con_sq = UMAX**2 - ca.mtimes([con, con.T])
            opti.subject_to(con_sq >= 0)

        opts_setting = {'ipopt.max_iter': 1e5,
                        'ipopt.print_level': 0,
                        'print_time': 0}
        opti.solver('ipopt', opts_setting)

        # cost function
        obj = self.costFunction(opt_states, opt_controls, self.goal, scan_data)
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

    def costFunction(self, opt_states, opt_controls, goal, scan_data):

        c_u = self.costControl(opt_controls)
        c_tra = self.costTracking(opt_states, goal)
        c_col = self.costCollision(opt_states, scan_data)
        total = c_tra + c_u + c_col

        return total

    def costControl(self, u):
        cost_u = 0
        for i in range(HORIZON_LENGTH):
            control = u[i,:]
            cost_u = ca.mtimes(control, control.T)
        return W_u*cost_u

    def costTracking(self, traj, goal):
        cost_tra = 0
        for i in range(HORIZON_LENGTH):
            pos_rel = traj[i,:3].T - goal
            cost_tra += ca.mtimes(pos_rel.T, pos_rel)
        return W_tra*cost_tra
    
    def costCollision(self, traj, scan_data):
        cost_col = 0
        ang, dist = scan_data
        np.min
        if dist.shape[0] != 0:
            num = len(dist)
            for j in range(1,num,2):
                obs_x = dist[j] * np.cos(ang[j]) + self.state[0]
                obs_y = dist[j] * np.sin(ang[j]) + self.state[1]
                for i in range(HORIZON_LENGTH):
                    obs_rel = traj[i,:2].T - np.array([obs_x, obs_y])
                    cost_col += 1./(1+ca.exp(4*(ca.mtimes(obs_rel.T, obs_rel) - ROBOT_RADIUS)))
        return W_col*cost_col