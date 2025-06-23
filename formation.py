import numpy as np
def generate_circular_formation(num_drones, radius, arc_angle_deg=180):
    """
    Generates offset vectors for a circular formation.

    Args:
        num_drones (int): The number of drones in the formation.
        radius (float): The radius of the circle.
        initial_angle_deg (float): The angle of the first drone in degrees,
                                   relative to the positive x-axis.

    Returns:
        np.array: An array of shape (num_drones, 3) containing the [x, y, z] offsets.
    """
    if num_drones == 0:
            return np.array([])
    offsets = []
    leader_offset = [0.0, 0.0, 0.0]
    offsets.append(leader_offset)
    
    if num_drones == 1:
        return np.array(offsets)
    
    num_followers = num_drones - 1
    arc_angle_rad = np.deg2rad(arc_angle_deg)
    start_angle_rad = -arc_angle_rad / 2.0
    if num_followers == 1:
        angle_step_rad = 0
        start_angle_rad = 0 
    else:
        angle_step_rad = arc_angle_rad / (num_followers - 1)
    for i in range(num_followers):
        angle = start_angle_rad + i * angle_step_rad
        
        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)
        offset_z = 0.0
        
        offsets.append([offset_x, offset_y, offset_z])
        
    return np.array(offsets)