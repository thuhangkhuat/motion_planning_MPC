import numpy as np

def calculate_fov_corners(state, hfov, vfov, prev_yaw=0.0):
    """
    Calculate the corners of the Field of View (FOV) for a robot based on its state.
    """
    position = state[:3]  
    velocity_2d = state[3:5] #

    x, y, height = position
    vx, vy = velocity_2d
    
    if np.linalg.norm(velocity_2d) > 1e-5:
        yaw = np.arctan2(vy, vx)
    else:
        yaw = prev_yaw
    hfov_rad = np.deg2rad(hfov)
    vfov_rad = np.deg2rad(vfov)

    width_half = height * np.tan(hfov_rad / 2.0)
    length_half = height * np.tan(vfov_rad / 2.0)

    local_corners = np.array([
        [+length_half, +width_half],  
        [+length_half, -width_half],  
        [-length_half, -width_half],  
        [-length_half, +width_half] 
    ])

    c, s = np.cos(yaw), np.sin(yaw)
    rotation_matrix = np.array([[c, -s], [s,  c]])
    rotated_corners = local_corners @ rotation_matrix.T
    world_corners = rotated_corners + np.array([x, y])
    fov_corners_to_plot = np.vstack([world_corners, world_corners[0]])

    return fov_corners_to_plot, width_half * 2, length_half * 2, yaw


if __name__ == '__main__':
    test_robot_state = np.array([5, 5, 3, 1.0, 1.0, 0.0]) 

    HFOV = 80.0
    VFOV = 60.0
    
    previous_yaw = 0.0

    corners, width, length, calculated_yaw = calculate_fov_corners(
        test_robot_state, 
        HFOV, VFOV,
        prev_yaw=previous_yaw
    )

    if corners is not None:
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 8))
        plt.plot(corners[:, 0], corners[:, 1], 'r-', label="FOV")
        plt.fill(corners[:-1, 0], corners[:-1, 1], 'r', alpha=0.2)
        
        plt.scatter(test_robot_state[0], test_robot_state[1], c='b', marker='o', s=100, label="Robot Position")

        arrow_length = 1.0
        plt.arrow(
            test_robot_state[0], test_robot_state[1], 
            arrow_length * np.cos(calculated_yaw), arrow_length * np.sin(calculated_yaw), 
            head_width=0.2, head_length=0.3, fc='b', ec='b'
        )

        plt.xlabel("X"); plt.ylabel("Y")
        plt.title("FOV with Yaw derived from 6D State")
        plt.grid(True); plt.axis('equal'); plt.legend(); plt.show()