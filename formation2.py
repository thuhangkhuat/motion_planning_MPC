import numpy as np

def generate_arc_formation(num_drones, radius, arc_angle_deg=180.0):
    """
    Generates offset vectors for drones distributed evenly on an arc.
    The arc is centered symmetrically in front (along the positive x-axis).

    Args:
        num_drones (int): The number of drones to place on the arc.
        radius (float): The radius of the arc.
        arc_angle_deg (float): The total angle of the arc in degrees.

    Returns:
        np.array: An array of shape (num_drones, 3) for the offsets.
                  Returns an empty array if num_drones is 0.
    """
    if num_drones <= 0:
        return np.array([])
        
    offsets = []
    arc_angle_rad = np.deg2rad(arc_angle_deg)
    
    # Calculate the starting angle and the angle step between drones
    # The starting angle is -half the arc angle to center it.
    start_angle_rad = -arc_angle_rad / 2.0
    
    # Handle the special case of a single drone on the arc
    if num_drones == 1:
        # Place it directly in front (angle = 0)
        angle_step_rad = 0
        start_angle_rad = 0
    else:
        # The angle step is the total arc angle divided by the number of gaps
        angle_step_rad = arc_angle_rad / (num_drones - 1)

    # Generate offsets for each drone
    for i in range(num_drones):
        angle = start_angle_rad + i * angle_step_rad
        
        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)
        offset_z = 0.0
        
        offsets.append([offset_x, offset_y, offset_z])
        
    return np.array(offsets)


def generate_leader_arc_formation(num_drones, radius, arc_angle_deg=180.0):
    """
    Generates offsets for a formation with one leader at the center (target)
    and the rest forming an arc in front.

    Args:
        num_drones (int): The total number of drones in the formation.
        radius (float): The radius of the arc for the follower drones.
        arc_angle_deg (float): The total angle of the arc.

    Returns:
        np.array: An array of shape (num_drones, 3) for the offsets.
    """
    if num_drones <= 0:
        return np.array([])
    
    # 1. Define the leader's offset (at the center)
    leader_offset = np.array([[0.0, 0.0, 0.0]])

    if num_drones == 1:
        return leader_offset
    
    # 2. Generate offsets for the follower drones
    num_followers = num_drones - 1
    follower_offsets = generate_arc_formation(num_followers, radius, arc_angle_deg)
    
    # 3. Combine the leader and follower offsets
    # np.vstack stacks arrays vertically
    full_formation_offsets = np.vstack([leader_offset, follower_offsets])
    
    return full_formation_offsets

if __name__ == '__main__':
    # Test hàm generate_leader_arc_formation
    num_agents = 5
    formation_radius = 4.0
    arc_angle = 270.0

    offsets = generate_leader_arc_formation(num_agents, formation_radius, arc_angle)

    print(f"--- Leader-Arc Formation with {num_agents} drones ---")
    print(offsets)
    # Expected output:
    # First row should be [0, 0, 0]
    # The other 4 rows should be the offsets for the followers on the arc

    # Test hàm generate_arc_formation (hàm cơ sở)
    arc_offsets = generate_arc_formation(4, formation_radius, arc_angle)
    print("\n--- Arc-Only Formation with 4 drones ---")
    print(arc_offsets)