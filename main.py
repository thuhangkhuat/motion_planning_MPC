
from config import *
if METHOD == 1:
    from robot import Robot

import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from target_rrt import SequentialRRTPlanner
from target import Target


if __name__ == "__main__":

    # Initialize target
    target = SequentialRRTPlanner(TAR_WAYPOINTS)
    target.generateTrajectory()
    target_traj = []
    robots = []

    # Initialize Robot
    for i in range(NUM_ROBOT):
        robot = Robot(i, np.concatenate([STARTS[i,:],[0,0,0]]), GOALS[i,:])
        robots.append(robot)
    
    compute_times = []
    iter = 0
    try:
        print("[INFO] Start")
        while True:
            # Update target
            target.update()
            target_traj.append(target.state.copy())
            # compute velocity using nmpc
            start = time.time()
            for i in range(NUM_ROBOT):
                robots[i].goal = target.state.copy()
                robots[i].computeControlSignal(robots)
                compute_times.append(time.time()-start)
            iter += 1
            if iter % 10 == 0:
                print("Iteration {}".format(iter))
                
            distance_to_final_dest = np.linalg.norm(target.state - target.final_destination)
            if distance_to_final_dest < 0.3: 
                print(f"[INFO] Target has reached its final destination. Stopping simulation.")
                break
    finally:
        print("[INFO] Saving")
        # Saving
        data = {}
        for i in range(NUM_ROBOT):
            r = {}
            r["path"] = np.array(robots[i].path)
            r["traj_refs"] = np.array(robots[i].traj_refs)
            r["tar_traj"] = np.array(target_traj)
            r["corridors"] = np.array(robots[i].corridors)
            data[i] = r
        with open(FILE_NAME, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        compute_times = np.array(compute_times)
        print("Average time: {:.6}s".format(compute_times.mean()))
        print("Max time: {:.6}s".format(compute_times.max()))   
        print("Min time: {:.6}s".format(compute_times.min()))
