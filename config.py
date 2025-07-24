import numpy as np
import time
import matplotlib.pyplot as plt
from formation import generate_circular_formation
from formation3 import generate_safe_leader_arc_formation

TIMESTEP = 0.1
ROBOT_RADIUS = 0.25
SENSING_RADIUS = 3.0
SENSING_NEIGHBOR = 5.0 
EPSILON = 0.1
D_FRAC = 0.0
GRID_SIZE = 0.1
EXPAND_SIZE = 3

VMAX = 1.0
UMAX = 5.0

HORIZON_LENGTH = 10

METHOD = 1  # 1 - our, 2 - mpc, 3 - apf

SCENARIO = 5
if SCENARIO == 1:
    #Parameters of target
    TAR_MAX_SPEED = 0.8
    TAR_WAYPOINTS = [np.array([4.0, 5.0, 5.0]),np.array([21.0, 5.0, 5.0])]
    TAR_EPSILON = 0.1

    # Parameters of the environment
    VIEWING_RADIUS = 2.5    # R_view: radius of the viewing area
    MIN_SEPARATION = 1.0    # d_min: minimum distance between robots
    MAX_SEPARATION = 8.0    # d_max: maximum distance between robots

    FORMATION_OFFSETS = np.array([
    [0.0, 0.0, 0.0],                          # Drone 0
    [VIEWING_RADIUS * 0.5,  VIEWING_RADIUS * 0.5, 0.0], # Drone 1
    [VIEWING_RADIUS * 0.5, -VIEWING_RADIUS * 0.5, 0.0]  # Drone 2
    ])
    # Parameters of FOV
    HFOV = 60.0             # Horizontal field of view
    VFOV = 80.0             # Vertical field of view

    # Weights for MPC
    W_tra = 0.5
    W_u = 4e-1
    W_col = 1.5
    W_slack = 6.0
    W_form_dist = 3.0
    W_form_struct = 1.0

    # Weights for CBF
    DT_CBF_GAMMA = 0.5 
    STARTS = np.array([[2.5, 4., 3.],
                       [2.5, 5., 3.],
                       [2.5, 3., 3.],])
    NUM_ROBOT = STARTS.shape[0]
    GOALS = STARTS + np.array([21., 0., 0.])
    
    # Obstacle x, y, r
    OBSTACLES = np.array([[ 7.0, 2.5, 0.8],
                          [ 7.0, 6.7, 0.8],
                          [ 7.0, 9.0, 0.8],
                          [ 9.0, 1.5, 0.8],
                          [12.0, 3.7, 0.8],
                          [12.0, 7.5, 0.8],
                          [15.0, 1.5, 0.8],
                          [15.0, 6.0, 0.8],
                          [17.0, 9.0, 0.8],
                          [19.0, 3.5, 0.8],
                          [19.0, 6.5, 0.8]])
    # OBSTACLES = np.array([])
    XLIM = [0, 25]
    YLIM = [0, 10]
elif SCENARIO == 2:
    #Parameters of target
    TAR_MAX_SPEED = 0.8
    TAR_WAYPOINTS = [np.array([3.0, 5.0, 5.0]),np.array([10.0, 7.0, 5.0]),
                     np.array([15.0, 3.0, 5.0]),np.array([21.0, 5.0, 5.0])]
    # TAR_WAYPOINTS = [np.array([3.0, 5.0, 5.0]),np.array([21.0, 5.0, 5.0])]
    TAR_EPSILON = 0.1

    # Parameters of the environment
    VIEWING_RADIUS = 2.5    # R_view: radius of the viewing area
    DESIRED_SEPARATION = 1.5   # desired distance between robots
    OFFSET_SEPARATION = 0.5    # offset distance between robots
    # Parameters of FOV
    HFOV = 60.0             # Horizontal field of view
    VFOV = 80.0             # Vertical field of view

    # Weights for MPC
    W_tra = 5.0
    W_u = 4e-1
    W_col = 1
    W_slack = 100.0
    W_form_dist = 2.0
    W_form_struct = 0.0

    # Weights for CBF
    DT_CBF_GAMMA = 0.5 
    STARTS = np.array([[1., 5., 3.],
                       [1., 4., 3.],
                       [1., 6., 3.]])
                    #    [1., 6.5, 3.]
                    #    [1., 7., 3.]])
    NUM_ROBOT = STARTS.shape[0]
    FORMATION_OFFSETS = generate_circular_formation(NUM_ROBOT, VIEWING_RADIUS -1 , arc_angle_deg=60)

    GOALS = STARTS + np.array([21., 0., 0.])
    
    # Obstacle x, y, r
    OBSTACLES = np.array([[ 7.0, 2.7, 0.8],
                          [ 7.0, 5.8, 0.8],
                          [ 7.0, 9.0, 0.8],
                          [ 9.0, 1.5, 0.8],
                          [12.0, 3.5, 0.8],
                          [12.0, 7.5, 0.8],
                          [15.0, 3.1, 0.8],
                          [15.0, 6.4, 0.8],
                          [17.0, 9.0, 0.8],
                          [19.0, 3.2, 0.8],
                          [19.0, 6.6, 0.8]])
    # OBSTACLES = np.array([])
    XLIM = [0, 25]
    YLIM = [0, 10]
elif SCENARIO == 3:
    #Parameters of target
    TAR_MAX_SPEED = 0.8
    TAR_WAYPOINTS = [np.array([5.0, 2.0, 5.0]),np.array([10.0, 7.0, 5.0]),
                     np.array([15.0, 5.0, 5.0]),np.array([21.0, 5.0, 5.0])]
    TAR_EPSILON = 0.1

    VIEWING_RADIUS = 2.5    # R_view: radius of the viewing area
    DESIRED_SEPARATION = 1.5   # desired distance between robots
    
    # Parameters of FOV
    HFOV = 60.0             # Horizontal field of view
    VFOV = 80.0             # Vertical field of view
    # Weights for MPC
    W_tra = 3.0
    W_u = 4e-1
    W_col = 2.5
    W_slack = 10.0
    W_form_dist = 3
    W_form_struct = 0

    # Weights for CBF
    DT_CBF_GAMMA = 0.5
    STARTS = np.array([[1.5, 5., 3.],
                       [1.5, 4., 3.],
                       [1.5, 6., 3.],])
    # STARTS = np.array([[2., 3., 5.]])
    GOALS = STARTS + np.array([21., 0., 0.])
    NUM_ROBOT = STARTS.shape[0]
    FORMATION_OFFSETS = generate_circular_formation(NUM_ROBOT, VIEWING_RADIUS - 1, arc_angle_deg=60)
    # Obstacle x, y, r
    OBSTACLES = np.array([[ 7.0, 2.5, 0.8],
                          [ 7.0, 5.9, 0.8],
                          [ 7.0, 9.0, 0.8],
                          [ 9.0, 1.5, 0.8],
                          [12.0, 4.0, 0.8],
                          [12.0, 7.5, 0.8],
                          [15.0, 1.5, 0.8],
                          [15.0, 6.0, 0.8],
                          [17.0, 9.0, 0.8],
                          [19.0, 3.5, 0.8],
                          [19.0, 6.5, 0.8]])
    XLIM = [0, 25]
    YLIM = [0, 10]

elif SCENARIO == 4:
    #Parameters of target
    TAR_MAX_SPEED = 0.8
    TAR_WAYPOINTS = [np.array([2.5, 5.0, 5.0]),np.array([10.0, 7.0, 5.0]),
                     np.array([15.0, 5.0, 5.0]),np.array([18.0, 10.0, 5.0]),np.array([25.0, 5.0, 5.0])]
    # TAR_WAYPOINTS = [np.array([3.0, 5.0, 5.0]),np.array([22.0, 5.0, 5.0])]
    TAR_EPSILON = 0.1

    VIEWING_RADIUS = 3.5    # R_view: radius of the viewing area
    DESIRED_SEPARATION = 2   # desired distance between robots
    
    # Parameters of FOV
    HFOV = 60.0             # Horizontal field of view
    VFOV = 80.0             # Vertical field of view
    # Weights for MPC
    W_tra = 5.0
    W_u = 4e-1
    W_col = 5.0
    W_slack = 20
    W_form_dist = 5.0
    W_form_struct = 0

    # Weights for CBF
    DT_CBF_GAMMA = 0.5
    STARTS = np.array([[0.5, 5, 3.],
                       [0.5, 3, 3.],
                       [0.5, 4, 3.],
                       [0.5, 6, 3.],
                       [0.5, 7, 3.]])
    # STARTS = np.array([[2., 3., 5.]])
    GOALS = STARTS + np.array([21., 0., 0.])
    NUM_ROBOT = STARTS.shape[0]
    FORMATION_OFFSETS = generate_circular_formation(NUM_ROBOT, VIEWING_RADIUS-0.9, arc_angle_deg=180)
    # Obstacle x, y, r
    # OBSTACLES = np.array([[ 7.0, 2.5, 0.8],
    #                       [ 7.0, 5.9, 0.8],
    #                       [ 7.0, 9.0, 0.8],
    #                       [ 9.0, 1.5, 0.8],
    #                       [12.0, 4.0, 0.8],
    #                       [12.0, 7.5, 0.8],
    #                       [15.0, 1.5, 0.8],
    #                       [15.0, 6.0, 0.8],
    #                       [17.0, 9.0, 0.8],
    #                       [19.0, 3.5, 0.8],
    #                       [19.0, 6.5, 0.8]])
    OBSTACLES = np.array([])
    XLIM = [0, 27]
    YLIM = [0, 14]

elif SCENARIO == 5:
    #Parameters of target
    TAR_MAX_SPEED = 0.8
    TAR_WAYPOINTS = [np.array([3, 5, 5.0]),np.array([10.0, 7.0, 5.0]),
                     np.array([15.0, 7.0, 5.0]),np.array([21.0, 5.0, 5.0])]
    # TAR_WAYPOINTS = [np.array([2.5, 5.0, 5.0]),np.array([21.0, 5.0, 5.0])]
    TAR_EPSILON = 0.1

    VIEWING_RADIUS = 2.5    # R_view: radius of the viewing area
    DESIRED_SEPARATION = VIEWING_RADIUS   # desired distance between robots
    
    # Parameters of FOV
    HFOV = 60.0             # Horizontal field of view
    VFOV = 60.0             # Vertical field of view
    # Weights for MPC
    W_tra = 1
    W_u = 4e-1
    W_col = 2.0
    W_slack = 4.0
    W_form_dist = 10.0
    W_form_spread = 20.0

    # Weights for CBF
    DT_CBF_GAMMA = 0.5
    STARTS = np.array([[1, 5, 3.],])
                    #    [4, 1, 3.],
                    #    [2, 11, 3.],])
    # STARTS = np.array([[2., 3., 5.]])
    GOALS = STARTS + np.array([21., 0., 0.])
    NUM_ROBOT = STARTS.shape[0]
    # FORMATION_OFFSETS = generate_circular_formation(NUM_ROBOT, VIEWING_RADIUS-0.5, arc_angle_deg=120)
    FORMATION_OFFSETS = generate_safe_leader_arc_formation(NUM_ROBOT, VIEWING_RADIUS, DESIRED_SEPARATION)
    # Obstacle x, y, r
    OBSTACLES = np.array([[ 7.0, 2.5, 0.5],
                          [ 7.0, 5.0, 0.5],
                          [ 7.0, 9.0, 0.5],
                          [ 9.0, 1.5, 0.5],
                          [12.0, 4.0, 0.5],
                          [12.0, 9.5, 0.5],
                          [15.0, 1.5, 0.5],
                          [15.0, 4.5, 0.5],
                          [17.0, 9.0, 0.5],
                          [19.0, 3.5, 0.5],
                          [19.0, 6.5, 0.5]])
    # OBSTACLES = np.array([])
    # OBSTACLES = np.array([])
    XLIM = [0, 22]
    YLIM = [0, 12]

FILE_NAME = "data{}_scen{}_{}.txt".format(METHOD, SCENARIO, NUM_ROBOT)
SAVE_GIF = "results/data{}_scen{}_{}.gif".format(METHOD, SCENARIO, NUM_ROBOT)