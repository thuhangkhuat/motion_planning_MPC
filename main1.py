
from config import *
if METHOD == 1:
    from robot import Robot

import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import target as Target


if __name__ == "__main__":

    # Initialize target
    target = Target.Target(TAR_WAYPOINTS)
    target.generateTrajectory()
    target_traj = []
    robots = []
    previous_target_state = target.state.copy()
    target_yaw = 0.0
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
            target_velocity = target.state - previous_target_state
            previous_target_state = target.state.copy()
            if np.linalg.norm(target_velocity[:2]) > 1e-6:
                target_yaw = np.arctan2(target_velocity[1], target_velocity[0])
            c, s = np.cos(target_yaw), np.sin(target_yaw)
            rotation_matrix = np.array([[c, -s], 
                                        [s,  c]])
            # compute velocity using nmpc
            start = time.time()
            for i in range(NUM_ROBOT):
                base_offset = FORMATION_OFFSETS[i]
                base_offset_2d = base_offset[:2]
                rotated_offset_2d = rotation_matrix @ base_offset_2d
                rotated_offset = np.array([rotated_offset_2d[0], rotated_offset_2d[1], base_offset[2]])
                # print(f"[INFO] Robot {i} offset vector: {offset_vector}")
                virtual_target_pos = target.state.copy()  + rotated_offset
                robots[i].goal = virtual_target_pos
                robots[i].computeControlSignal(robots)
                compute_times.append(time.time()-start)
            iter += 1
            if iter % 10 == 0:
                print("Iteration {}".format(iter))

            # Reach terminal condition
            # count = 0
            # for i in range(NUM_ROBOT):
            #     if np.linalg.norm(robots[i].state[:3] - robots[i].goal) < EPSILON:
            #         count += 1
            # if count == NUM_ROBOT:
            #     break
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
            data[i] = r
        with open(FILE_NAME, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        compute_times = np.array(compute_times)
        print("Average time: {:.6}s".format(compute_times.mean()))
        print("Max time: {:.6}s".format(compute_times.max()))   
        print("Min time: {:.6}s".format(compute_times.min()))
