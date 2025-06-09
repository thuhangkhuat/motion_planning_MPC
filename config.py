import numpy as np
import time
import matplotlib.pyplot as plt

TIMESTEP = 0.1
ROBOT_RADIUS = 0.25
SENSING_RADIUS = 3.0
EPSILON = 0.1
D_FRAC = 0.0
GRID_SIZE = 0.1
EXPAND_SIZE = 3

VMAX = 1.0
UMAX = 5.0

HORIZON_LENGTH = 10

METHOD = 1  # 1 - our, 2 - mpc, 3 - apf

#Parameters of target
TAR_MAX_SPEED = 1
TAR_STARTS = np.array([5.0, 10.0, 5.0])  
TAR_GOALS = np.array([21.0, 5.0, 5.0])  
TAR_EPSILON = 0.1

W_tra = 1.0
W_u = 4e-1
W_col = 1.5

SCENARIO = 1
if SCENARIO == 1:
    STARTS = np.array([[2., 3., 5.],
                       [2., 5., 5.],
                       [2., 7., 5.],])
    GOALS = STARTS + np.array([21., 0., 0.])
    NUM_ROBOT = STARTS.shape[0]

    # Obstacle x, y, r
    OBSTACLES = np.array([[ 7.0, 2.5, 0.8],
                          [ 7.0, 6.0, 0.8],
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
elif SCENARIO == 2:
    CR = 5
    NUM_ROBOT = 4
    STARTS = []; GOALS = []
    for i in range(NUM_ROBOT):
        STARTS.append(np.array([CR*np.cos(2*np.pi*i/NUM_ROBOT),
                                CR*np.sin(2*np.pi*i/NUM_ROBOT),
                                5.0]))
        GOALS.append(np.array([-CR*np.cos(2*np.pi*i/NUM_ROBOT),
                               -CR*np.sin(2*np.pi*i/NUM_ROBOT),
                                5.0]))
    STARTS = np.array(STARTS); GOALS = np.array(GOALS)
    # Obstacle x, y, r
    # OBSTACLES = np.array([])
    OBSTACLES = np.array([[ 0.0, 0.5, 0.5],
                          [ 1.0,-2.0, 0.5],
                          [-1.0, 3.0, 0.5],
                          [ 3.0,-1.5, 0.5],
                          [-3.0, 1.0, 0.5],])

    XLIM = [-CR-1., CR+1]
    YLIM = [-CR-1., CR+1]

FILE_NAME = "data{}_scen{}_{}.txt".format(METHOD, SCENARIO, NUM_ROBOT)
SAVE_GIF = "results/data{}_scen{}_{}.gif".format(METHOD, SCENARIO, NUM_ROBOT)