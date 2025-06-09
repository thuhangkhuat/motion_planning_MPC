
from config import *
if METHOD == 1:
    from robot import Robot
elif METHOD == 2:
    from robot_mpc import Robot
elif METHOD == 3:
    from robot_apf import Robot

import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":

    robots = []
    # Initialize Robot
    for i in range(NUM_ROBOT):
        robot = Robot(0, np.concatenate([STARTS[i,:],[0,0,0]]), GOALS[i,:])
        robots.append(robot)
    
    compute_times = []
    iter = 0
    try:
        print("[INFO] Start")
        while True:
            # compute velocity using nmpc
            start = time.time()
            for i in range(NUM_ROBOT):
                robots[i].computeControlSignal(robots)
                compute_times.append(time.time()-start)
            iter += 1
            if iter % 10 == 0:
                print("Iteration {}".format(iter))

            # Reach terminal condition
            count = 0
            for i in range(NUM_ROBOT):
                if np.linalg.norm(robots[i].state[:3] - robots[i].goal) < EPSILON:
                    count += 1
            if count == NUM_ROBOT:
                break
    finally:
        print("[INFO] Saving")
        # Saving
        data = {}
        for i in range(NUM_ROBOT):
            r = {}
            r["path"] = np.array(robots[i].path)
            r["traj_refs"] = np.array(robots[i].traj_refs)
            data[i] = r
        with open(FILE_NAME, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        compute_times = np.array(compute_times)
        print("Average time: {:.6}s".format(compute_times.mean()))
        print("Max time: {:.6}s".format(compute_times.max()))   
        print("Min time: {:.6}s".format(compute_times.min()))
