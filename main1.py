from config import *
if METHOD == 1:
    from robot_rrt import Robot
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from target_cache import load_or_generate_target

# Đặt True để buộc gen lại target trajectory (vd: khi đổi waypoints/scenario)
FORCE_REGEN_TARGET = False


if __name__ == "__main__":

    # Initialize target (load from cache or generate)
    target = load_or_generate_target(TAR_WAYPOINTS, force_regen=FORCE_REGEN_TARGET)
    target_traj = []
    robots = []

    # Initialize Robot
    for i in range(NUM_ROBOT):
        robot = Robot(i, np.concatenate([STARTS[i, :], [0, 0, 0]]), GOALS[i, :])
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
            for i in range(NUM_ROBOT):
                t0 = time.time()             # đo time đúng cho từng UAV
                robots[i].goal = target.state.copy()
                robots[i].computeControlSignal(robots)
                compute_times.append(time.time() - t0)

            iter += 1
            if iter % 10 == 0:
                print("Iteration {}".format(iter))

            distance_to_final_dest = np.linalg.norm(
                target.state - target.final_destination)
            if distance_to_final_dest < 0.3:
                print("[INFO] Target has reached its final destination. "
                      "Stopping simulation.")
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
            r["corridors"] = np.array(robots[i].corridors_plot)
            data[i] = r
        with open(FILE_NAME, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        compute_times = np.array(compute_times)
        if len(compute_times) > 0:
            print("Average time: {:.6}s".format(compute_times.mean()))
            print("Max time: {:.6}s".format(compute_times.max()))
            print("Min time: {:.6}s".format(compute_times.min()))