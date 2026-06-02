import numpy as np
import time
import matplotlib.pyplot as plt


# RRT parameters for uav
STEP_LENGTH = 0.1
GOAL_SAMPLE_RATE = 0.01
MAX_ITER = 5000

# Parameters for uav
TIMESTEP = 0.1
ROBOT_RADIUS = 0.3
SENSING_RADIUS = 10.0
SENSING_NEIGHBOR = 5.0 
EPSILON = 0.1
D_FRAC = 0.0
VMAX = 5
UMAX = 20
HORIZON_LENGTH = 10

# RRT parameters for target
TAR_STEP_LENGTH = 0.5
TAR_GOAL_SAMPLE_RATE = 0.1
TAR_MAX_ITER = 5000
TAR_RADIUS = 0.2
SAFETY_MARGIN = 0.4

METHOD = 2  # 1: A*, 2: JPS, 3: RRT

SCENARIO = 5
if SCENARIO == 1:
    #Parameters of target
    TAR_MAX_SPEED = 6.5
    TAR_WAYPOINTS = [np.array([40.0, 50.0,0]),np.array([300.0, 300.0,0]), np.array([400.0, 80.0,0])]
    TAR_EPSILON = 0.1

    # Parameters of the environment
    VIEWING_RADIUS = 30    # R_view: radius of the viewing area
    DESIRED_SEPARATION = VIEWING_RADIUS   # desired distance between robots

    # Parameters of FOV
    HFOV = 90.0             # Horizontal field of view
    VFOV = 90.0             # Vertical field of view

    # Weights for MPC
    W_tra = 10
    W_gui = 1
    W_u = 4e-1
    W_slack = 10.0
    W_form_dist = 5.0
    W_form_spread = 5.0
    W_col = 0.0
    W_corridor = 10

    # Weights for CBF
    DT_CBF_GAMMA = 0.5 
    STARTS = np.array([[40, 40, 3.],
                       [40, 30, 3.],
                       [30, 40, 3.],])
    NUM_ROBOT = STARTS.shape[0]
    GOALS = STARTS + np.array([21., 0., 0.])
    
     # Obstacle x, y, r
    POLYGON_OBSTACLES = [
    # np.array([[50.0, 50.0], [80.0, 50.0], [80.0, 80.0], [50.0, 80.0]]),
    np.array([[100.0, 120.0], [130.0, 120.0], [130.0, 160], [100, 160]]),
    np.array([[200,220], [230, 220], [230, 290], [220, 290]]),
    np.array([[300,100], [360, 100], [360, 140], [300, 140]]),
    np.array([[170,80], [220, 80], [220, 110], [170, 110]]),
    np.array([[345,235], [405, 235], [405, 275], [345, 275]]),
    np.array([[380,350], [420, 350], [420, 410], [380, 410]]),
    np.array([[170,340], [220, 340], [220, 370], [170, 370]]),
    np.array([[70,370], [100, 370], [100, 420], [70, 420]]),
    np.array([[220,430], [250, 430], [250, 460], [220, 460]]),
    np.array([[50,240], [90, 240], [90, 280], [50, 280]]),
    np.array([[390,35], [430, 35], [430, 75], [390, 75]]),]
  

    
    # RECTANGLE_OBSTACLES = [[10, 20, 20,20]] #x_min, y_min, width, height

    # OBSTACLES = np.array([[ 70.0, 20.5, 8],
    #                       [ 70.0, 60.7, 8],
    #                       [ 70.0, 90.0, 8],
    #                       [ 90.0, 10.5, 8],
    #                       [120.0, 30.7, 8],
    #                       [120.0, 70.5, 8],
    #                       [150.0, 10.5, 8],
    #                       [150.0, 60.0, 8],
    #                       [170.0, 90.0, 8],
    #                       [190.0, 30.5, 8],
    #                       [190.0, 60.5, 8]])

    RECTANGLE_OBSTACLES = [[50, 50, 30,30], 
                         [100, 120,30, 40],
                         [200,200,30, 70],
                         [300,100,60, 40],
                         [170,80,50,30],
                         [345,235, 60,40],
                         [380,350,40,60],
                         [170,340,50,30],
                         [70,370,30,50],
                         [220,430,30,30],
                         [50,240,40,40],
                         [390,35,40,40]]
    

    # RECTANGLE_OBSTACLES = []
    OBSTACLES = np.array([])
    XLIM = [0, 500]
    YLIM = [0, 500]
elif SCENARIO == 2:
    #Parameters of target
    TAR_MAX_SPEED = 8
    # TAR_WAYPOINTS = [np.array([16.0, 20.0,0]),np.array([120.0, 120.0,0]), np.array([160.0, 42.0,0])]
    # TAR_WAYPOINTS = [np.array([16.0, 20.0,0]),np.array([200.0, 175.0,0]),np.array([300.0, 175.0,0]),np.array([300.0, 400.0,0]),np.array([145.0, 400.0,0])]
    TAR_WAYPOINTS = [np.array([16.0, 20.0,0]),np.array([300.0, 300.0,0]), np.array([450.0, 150.0,0])]

    TAR_EPSILON = 0.1

    # Parameters of the environment
    VIEWING_RADIUS = 30    # R_view: radius of the viewing area
    DESIRED_SEPARATION = VIEWING_RADIUS   # desired distance between robots


    # Parameters of FOV
    HFOV = 90.0             # Horizontal field of view
    VFOV = 90.0             # Vertical field of view

    # Weights for MPC
    W_tra = 1
    W_gui = 1
    W_u = 4e-1
    W_slack = 10.0
    W_form_dist = 1.0
    W_form_spread = 1.0
    W_col = 0.0
    W_corridor = 10

    # Weights for CBF
    DT_CBF_GAMMA = 0.5 
    STARTS = np.array([[10, 10, 3.],
                       [50, 10, 3.],
                       [20, 50, 3.],])
    # STARTS = np.array([[10, 10, 3.],])
    NUM_ROBOT = STARTS.shape[0]
    GOALS = STARTS + np.array([21., 0., 0.])
    
     # Obstacle x, y, r
    POLYGON_OBSTACLES = [
    # np.array([[50.0, 50.0], [80.0, 50.0], [80.0, 80.0], [50.0, 80.0]]),
    np.array([[70.0, 120.0], [100.0, 120.0], [100.0, 170], [70, 170]]),
    np.array([[200,230], [230, 230], [230, 290], [200, 290]]),
    np.array([[300,100], [360, 100], [360, 140], [300, 140]]),
    np.array([[170,80], [220, 80], [220, 110], [170, 110]]),
    np.array([[345,235], [405, 235], [405, 275], [345, 275]]),
    np.array([[380,350], [420, 350], [420, 410], [380, 410]]),
    np.array([[170,340], [220, 340], [220, 370], [170, 370]]),
    np.array([[70,370], [100, 370], [100, 420], [70, 420]]),
    np.array([[220,430], [250, 430], [250, 460], [220, 460]]),
    np.array([[50,240], [90, 240], [90, 280], [50, 280]]),
    np.array([[390,35], [430, 35], [430, 75], [390, 75]]),]
  
    POLYGON_OBSTACLES = [poly * 1 for poly in POLYGON_OBSTACLES]

    
    # RECTANGLE_OBSTACLES = [[10, 20, 20,20]] #x_min, y_min, width, height

    # OBSTACLES = np.array([[ 70.0, 20.5, 8],
    #                       [ 70.0, 60.7, 8],
    #                       [ 70.0, 90.0, 8],
    #                       [ 90.0, 10.5, 8],
    #                       [120.0, 30.7, 8],
    #                       [120.0, 70.5, 8],
    #                       [150.0, 10.5, 8],
    #                       [150.0, 60.0, 8],
    #                       [170.0, 90.0, 8],
    #                       [190.0, 30.5, 8],
    #                       [190.0, 60.5, 8]])

    # RECTANGLE_OBSTACLES = [
    #                     #  [50, 50, 30,30], 
    #                      [70, 120,20, 50],
    #                      [200,230,30, 60],
    #                      [300,100,60, 40],
    #                      [170,80,50,30],
    #                      [345,235, 60,40],
    #                      [380,350,40,60],
    #                      [170,340,50,30],
    #                      [70,370,30,50],
    #                      [220,430,30,30],
    #                      [50,240,40,40],
    #                      [390,35,40,40]]
    RECTANGLE_OBSTACLES = [
                        #  [50, 50, 30,30], 
                         [70, 120,60, 70],
                         [200,230,50,80],
                         [300,100,80, 60],
                         [170,80,70,50],
                         [345,235, 90,70],
                         [380,350,60,80],
                         [170,340,70,50],
                         [70,370,50,70],
                         [220,430,50,50],
                         [50,240,60,60],
                         [390,35,60,60]]
    
    RECTANGLE_OBSTACLES = [np.array(rect) * 1 for rect in RECTANGLE_OBSTACLES]
    

    # RECTANGLE_OBSTACLES = []
    OBSTACLES = np.array([])
    XLIM = [0, 500]
    YLIM = [0, 500]
elif SCENARIO == 3:
    #Parameters of target
    TAR_MAX_SPEED = 8
    # TAR_WAYPOINTS = [np.array([214.0, 200.0,0]),np.array([300.0, 300.0,0])]
    # TAR_WAYPOINTS = [np.array([16.0, 20.0,0]),np.array([200.0, 175.0,0]),np.array([300.0, 175.0,0]),np.array([300.0, 400.0,0]),np.array([145.0, 400.0,0])]
    TAR_WAYPOINTS = [np.array([16.0, 20.0,0]),np.array([300.0, 300.0,0]), np.array([145.0, 465.0,0]),  np.array([450.0, 150.0,0])]

    TAR_EPSILON = 0.1

    # Parameters of the environment
    VIEWING_RADIUS = 30    # R_view: radius of the viewing area
    DESIRED_SEPARATION = VIEWING_RADIUS*1.5   # desired distance between robots


    # Parameters of FOV
    HFOV = 90.0             # Horizontal field of view
    VFOV = 90.0             # Vertical field of view

    # Weights for MPC
    W_tra = 1
    W_gui = 1
    W_u = 4e-1
    W_slack = 0.1
    W_form_dist = 0.1
    W_form_spread = 0.1
    W_col = 0.0
    W_corridor = 1.0
    DT_CBF_GAMMA_CORR = 0.3    # conservative cho corridor (safety)
    W_centroid = 0.3            # vừa, cân bằng formation
    W_corr_slack = 1e4          # rất cao - vi phạm = đụng obstacle

    # Weights for CBF
    DT_CBF_GAMMA = 0.5 
    STARTS = np.array([[20, 19, 3.],
                       [40, 50, 3.],
                       [0, 0, 3.],])
    # STARTS = np.array([[10, 10, 3.],])
    NUM_ROBOT = STARTS.shape[0]
    GOALS = STARTS + np.array([21., 0., 0.])
    
     # Obstacle x, y, r
    POLYGON_OBSTACLES = [
    # np.array([[50.0, 50.0], [80.0, 50.0], [80.0, 80.0], [50.0, 80.0]]),
    np.array([[70.0, 120.0], [100.0, 120.0], [100.0, 170], [70, 170]]),
    np.array([[200,230], [230, 230], [230, 290], [200, 290]]),
    np.array([[300,100], [360, 100], [360, 140], [300, 140]]),
    np.array([[170,80], [220, 80], [220, 110], [170, 110]]),
    np.array([[345,235], [405, 235], [405, 275], [345, 275]]),
    np.array([[380,350], [420, 350], [420, 410], [380, 410]]),
    np.array([[170,340], [220, 340], [220, 370], [170, 370]]),
    np.array([[70,370], [100, 370], [100, 420], [70, 420]]),
    np.array([[220,430], [250, 430], [250, 460], [220, 460]]),
    np.array([[50,240], [90, 240], [90, 280], [50, 280]]),
    np.array([[390,35], [430, 35], [430, 75], [390, 75]]),]
  
    POLYGON_OBSTACLES = [poly * 1 for poly in POLYGON_OBSTACLES]

    
    # RECTANGLE_OBSTACLES = [[10, 20, 20,20]] #x_min, y_min, width, height

    RECTANGLE_OBSTACLES = [
                        #  [50, 50, 30,30], 
                         [70, 120,40, 70],
                         [200,230,50,80],
                         [300,100,80, 60],
                         [170,80,70,50],
                         [345,235, 90,70],
                         [380,350,60,80],
                         [170,340,70,50],
                         [70,370,50,70],
                         [220,430,50,50],
                         [50,240,60,60],
                         [390,35,60,60]]
    
    RECTANGLE_OBSTACLES = [np.array(rect) * 1 for rect in RECTANGLE_OBSTACLES]
    

    # RECTANGLE_OBSTACLES = []
    OBSTACLES = np.array([])
    XLIM = [0, 500]
    YLIM = [0, 500]

elif SCENARIO == 4:
    #Parameters of target
    TAR_MAX_SPEED = 2
    # TAR_WAYPOINTS = [np.array([214.0, 200.0,0]),np.array([300.0, 300.0,0])]
    # TAR_WAYPOINTS = [np.array([16.0, 20.0,0]),np.array([200.0, 175.0,0]),np.array([300.0, 175.0,0]),np.array([300.0, 400.0,0]),np.array([145.0, 400.0,0])]
    TAR_WAYPOINTS = [np.array([3, 10,0]),np.array([30.0, 30.0,0]), np.array([42, 46,0]),  np.array([45.0, 15.0,0])]

    TAR_EPSILON = 0.1

    # Parameters of the environment
    VIEWING_RADIUS = 5    # R_view: radius of the viewing area
    DESIRED_SEPARATION = VIEWING_RADIUS*1.5   # desired distance between robots


    # Parameters of FOV
    HFOV = 90.0             # Horizontal field of view
    VFOV = 90.0             # Vertical field of view

    # Weights for MPC
    W_tra = 1
    W_gui = 1
    W_u = 4e-1
    W_slack = 0.1
    W_form_dist = 0.1
    W_form_spread = 0.1
    W_col = 0.0
    W_corridor = 1.0
    DT_CBF_GAMMA_CORR = 0.3    # conservative cho corridor (safety)
    W_centroid = 0.3            # vừa, cân bằng formation
    W_corr_slack = 1e4          # rất cao - vi phạm = đụng obstacle

    # Weights for CBF
    DT_CBF_GAMMA = 0.5 
    STARTS = np.array([[15.0, 30.0, 3.],
                       [4.0, 5.0, 3.],
                       [10.0, 5.0, 3.],
                       [20.0, 16.0, 3.],
                       [3.0, 20.0, 3.],
                       ])
    # STARTS = np.array([[10, 10, 3.],])
    NUM_ROBOT = STARTS.shape[0]
    GOALS = STARTS + np.array([21., 0., 0.])
    
     # Obstacle x, y, r
    POLYGON_OBSTACLES = [
    # np.array([[50.0, 50.0], [80.0, 50.0], [80.0, 80.0], [50.0, 80.0]]),
    np.array([[7.0, 12.0], [10.0, 12.0], [10.0, 17.0], [7.0, 17.0]]),
    np.array([[20.0,23.0], [23.0, 23.0], [23.0, 29.0], [20.0, 29.0]]),
    np.array([[30.0,10.0], [36.0, 10.0], [36.0, 14.0], [30.0, 14.0]]),
    np.array([[17.0,8.0], [22.0, 8.0], [22.0, 11.0], [17.0, 11.0]]),
    np.array([[34.5,23.5], [40.5, 23.5], [40.5, 27.5], [34.5, 27.5]]),
    np.array([[38.0,35.0], [42.0, 35.0], [42.0, 41.0], [38.0, 41.0]]),
    np.array([[17.0,34.0], [22.0, 34.0], [22.0, 37.0], [17.0, 37.0]]),
    np.array([[7.0,37.0], [10.0, 37.0], [10.0, 42.0], [7.0, 42.0]]),
    np.array([[22.0,43.0], [25.0, 43.0], [25.0, 46.0], [22.0, 46.0]]),
    np.array([[5.0,24.0], [9.0, 24.0], [9.0, 28.0], [5.0, 28.0]]),
    np.array([[39.0,3.5], [43.0, 3.5], [43.0, 7.5], [39.0, 7.5]]),
    ]
  

    RECTANGLE_OBSTACLES = [
                        #  [50, 50, 30,30], 
                         [7.0, 12.0,3.0, 5.0],
                         [20.0,23.0,5.0,8.0],
                         [30.0,10.0,8.0, 6.0],
                         [17.0,8.0,7.0,5.0],
                         [34.5,23.5, 9.0,7.0],
                         [38.0,35.0,6.0,8.0],
                         [17.0,34.0,7.0,5.0],
                         [7.0,37.0,5.0,7.0],
                         [22.0,43.0,5.0,5.0],
                         [5.0,24.0,6.0,6.0],
                         [39.0,3.5,6.0,6.0]
                         ]
    

    # RECTANGLE_OBSTACLES = []
    OBSTACLES = np.array([])
    XLIM = [0, 50]
    YLIM = [0, 50]

elif SCENARIO == 5:
    #Parameters of target
    TAR_MAX_SPEED = 2
    # TAR_WAYPOINTS = [np.array([214.0, 200.0,0]),np.array([300.0, 300.0,0])]
    # TAR_WAYPOINTS = [np.array([16.0, 20.0,0]),np.array([200.0, 175.0,0]),np.array([300.0, 175.0,0]),np.array([300.0, 400.0,0]),np.array([145.0, 400.0,0])]
    TAR_WAYPOINTS = [np.array([3, 10,0]),np.array([30.0, 30.0,0]), np.array([42, 46,0]),  np.array([45.0, 15.0,0])]

    TAR_EPSILON = 0.1

    # Parameters of the environment
    VIEWING_RADIUS = 5    # R_view: radius of the viewing area
    DESIRED_SEPARATION = VIEWING_RADIUS*1.5   # desired distance between robots


    # Parameters of FOV
    HFOV = 90.0             # Horizontal field of view
    VFOV = 90.0             # Vertical field of view

    # Weights for MPC
    W_tra = 1.5
    W_gui = 1
    W_u = 0.4
    W_slack = 0.1
    W_form_dist = 0.1
    W_form_spread = 0.1
    W_col = 0.0
    W_corridor = 1.0
    DT_CBF_GAMMA_CORR = 0.3    # conservative cho corridor (safety)
    W_centroid = 0.3            # vừa, cân bằng formation
    W_corr_slack = 1e4          # rất cao - vi phạm = đụng obstacle

    # Weights for CBF
    DT_CBF_GAMMA = 0.5 
    STARTS = np.array([[20.0, 20.0, 3.],
                       [4.0, 5.0, 3.],
                       [10.0, 5.0, 3.],
                    #    [15.0, 16.0, 3.],
                    #    [3.0, 14.0, 3.],
                       ])
    # STARTS = np.array([[10, 10, 3.],])
    NUM_ROBOT = STARTS.shape[0]
    GOALS = STARTS + np.array([21., 0., 0.])
    
     # Obstacle x, y, r
    POLYGON_OBSTACLES = [
    # np.array([[50.0, 50.0], [80.0, 50.0], [80.0, 80.0], [50.0, 80.0]]),
    np.array([[7.0, 12.0], [10.0, 12.0], [10.0, 17.0], [7.0, 17.0]]),
    np.array([[20.0,23.0], [23.0, 23.0], [23.0, 29.0], [20.0, 29.0]]),
    np.array([[30.0,10.0], [36.0, 10.0], [36.0, 14.0], [30.0, 14.0]]),
    np.array([[17.0,8.0], [22.0, 8.0], [22.0, 11.0], [17.0, 11.0]]),
    np.array([[34.5,23.5], [40.5, 23.5], [40.5, 27.5], [34.5, 27.5]]),
    np.array([[38.0,35.0], [42.0, 35.0], [42.0, 41.0], [38.0, 41.0]]),
    np.array([[17.0,34.0], [22.0, 34.0], [22.0, 37.0], [17.0, 37.0]]),
    np.array([[7.0,37.0], [10.0, 37.0], [10.0, 42.0], [7.0, 42.0]]),
    np.array([[22.0,43.0], [25.0, 43.0], [25.0, 46.0], [22.0, 46.0]]),
    np.array([[5.0,24.0], [9.0, 24.0], [9.0, 28.0], [5.0, 28.0]]),
    np.array([[39.0,3.5], [43.0, 3.5], [43.0, 7.5], [39.0, 7.5]]),]
  
    POLYGON_OBSTACLES = [poly * 1 for poly in POLYGON_OBSTACLES]

    RECTANGLE_OBSTACLES = [
                        #  [50, 50, 30,30], 
                         [7.0, 12.0,3.0, 5.0],
                         [20.0,23.0,5.0,8.0],
                         [30.0,10.0,8.0, 6.0],
                         [17.0,8.0,7.0,5.0],
                         [34.5,23.5, 9.0,7.0],
                         [38.0,35.0,6.0,8.0],
                         [17.0,34.0,7.0,5.0],
                         [7.0,37.0,5.0,7.0],
                         [22.0,43.0,5.0,5.0],
                         [5.0,24.0,6.0,6.0],
                         [39.0,3.5,6.0,6.0]]
    
    RECTANGLE_OBSTACLES = [np.array(rect) * 1 for rect in RECTANGLE_OBSTACLES]
    

    # RECTANGLE_OBSTACLES = []
    OBSTACLES = np.array([])
    XLIM = [0, 50]
    YLIM = [0, 50]

FILE_NAME = "data{}_scen{}_{}.txt".format(METHOD, SCENARIO, NUM_ROBOT)
FILE_NAME1 = "data{}_scen{}_{}.txt".format(METHOD, SCENARIO, NUM_ROBOT)
SAVE_GIF = "results/data{}_scen{}_{}.gif".format(METHOD, SCENARIO, NUM_ROBOT)