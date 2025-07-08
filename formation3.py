# formation.py
import numpy as np
import math

def generate_safe_leader_arc_formation(num_drones, viewing_radius, desired_separation):
    """
    Generates formation offsets that respect physical constraints.
    - One drone at the target (leader).
    - N-1 drones on an arc in front.
    - The arc radius and angle are calculated to satisfy viewing and separation constraints.

    Args:
        num_drones (int): Total number of drones.
        viewing_radius (float): The maximum allowed distance from the target.
        desired_separation (float): The desired distance between adjacent drones.

    Returns:
        np.array: An array of shape (num_drones, 3) for the offsets.
                  Returns None if the formation is impossible.
    """
    if num_drones <= 0:
        return np.array([])
    
    # Leader's offset is always at the center
    leader_offset = np.array([[0.0, 0.0, 0.0]])
    if num_drones == 1:
        return leader_offset

    # Determine a safe formation radius
    safety_buffer = 0.5
    formation_radius = viewing_radius - safety_buffer
    if formation_radius <= 0:
        print("ERROR: Viewing radius is too small for a safe formation.")
        return None

    # Calculate the required angle step between followers
    num_followers = num_drones - 1
    
    # Check if a solution is possible
    # The argument for arcsin must be between -1 and 1
    ratio = desired_separation / (2 * formation_radius)
    if ratio > 1.0:
        print(f"ERROR: Formation impossible. Drones can't maintain desired separation ({desired_separation}m) "
              f"within the given formation radius ({formation_radius}m).")
        print("Try increasing VIEWING_RADIUS or decreasing DESIRED_SEPARATION.")
        return None

    angle_step_rad = 2 * math.asin(ratio)

    # 4. Calculate the total arc angle for all followers
    total_arc_rad = angle_step_rad * (num_followers - 1)
    
    # 5. Generate offsets for followers
    start_angle_rad = -total_arc_rad / 2.0
    follower_offsets = []
    for i in range(num_followers):
        angle = start_angle_rad + i * angle_step_rad
        offset_x = formation_radius * np.cos(angle)
        offset_y = formation_radius * np.sin(angle)
        follower_offsets.append([offset_x, offset_y, 0.0])
        
    # 6. Combine and return
    return np.vstack([leader_offset, np.array(follower_offsets)])