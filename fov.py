import numpy as np
def calculate_fov_corners(state, hfov, vfov):
    """
    Calculate the corners of the field of view (FOV) based on the robot's state and FOV parameters.
    """
    x, y, height = state[:3]
    hfov_rad = np.deg2rad(hfov)
    vfov_rad = np.deg2rad(vfov)

    width_half = height * np.tan(hfov_rad / 2.0)
    length_half = height * np.tan(vfov_rad / 2.0)

    top_left = np.array([x - width_half, y + length_half])
    top_right = np.array([x + width_half, y + length_half])
    bottom_right = np.array([x + width_half, y - length_half])
    bottom_left = np.array([x - width_half, y - length_half])

    fov_corners = np.array([
    top_left,
    top_right,
    bottom_right,
    bottom_left,
    top_left
    ])

    return fov_corners, width_half *2, length_half *2


if __name__ == '__main__':
    test_robot_state = np.array([5, 5, 3])
    HFOV = 60.0
    VFOV = 80.0

    corners, width, length = calculate_fov_corners(test_robot_state, HFOV, VFOV)

    if corners is not None:
        print(corners, "Width:", width, "Length:", length)
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(corners[:, 0], corners[:, 1], 'r-') # Vẽ đa giác FOV
        plt.scatter(test_robot_state[0], test_robot_state[1], c='b', marker='o')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()