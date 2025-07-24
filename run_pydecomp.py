import math
import numpy as np
import matplotlib.pyplot as plt
import pydecomp as pdc  # Đổi tên import để giống với ví dụ của bạn

# ==============================================================================
# PHẦN 1: CODE CÓ SẴN CỦA BẠN (LIDAR VÀ CẤU HÌNH)
# Giữ nguyên không thay đổi
# ==============================================================================

# --- CONFIGURATION ---
OBSTACLES = np.array([
    [5, 5, 1.5], [3, 8, 1.0], [8, 12, 2.0],
    [12, 7, 1.2], [15, 15, 2.5], [7, 18, 1.8]
])
# OBSTACLES = np.array([
#     [5, 5, 1.5]
# ])
SENSING_RADIUS = 5.0

class LidarScanner:
    # ... (Toàn bộ code của lớp LidarScanner giữ nguyên ở đây) ...
    def __init__(self, range_min=0.1, range_max=10.0,
                            angle_min=-math.pi/2, angle_max = math.pi/2,
                            resolution=math.pi/180, noise=0.01):
        self.range_min = range_min
        self.range_max = range_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.resolution = resolution
        self.angle_num = int((self.angle_max - self.angle_min)/self.resolution) + 1
        self.range_num = int(10*range_max)
        self.noise = noise

    def distance(self, pose, obs_pose):
        ex = obs_pose[0] - pose[0]
        ey = obs_pose[1] - pose[1]
        return math.hypot(ex, ey)

    def isCollision(self, pose, obstacles):    
        for (cx, cy, cr) in obstacles:
            ex = cx - pose[0]; ey = cy - pose[1]
            if ex**2 + ey**2 - cr**2 <= 0:
                return True
        return False

    def senseObstacle(self, pose, obstacles_list):
        data = []
        x, y, theta = pose
        scan_angles = np.linspace(self.angle_min + theta, self.angle_max + theta, self.angle_num, True)
        for angle in scan_angles:
            sense = False
            x1 = x + self.range_min*math.cos(angle)
            y1 = y + self.range_min*math.sin(angle)
            x2 = x + self.range_max*math.cos(angle)
            y2 = y + self.range_max*math.sin(angle)
            for i in range(self.range_num+1):
                u = i/self.range_num
                x3 = x2*u + x1*(1 - u)
                y3 = y2*u + y1*(1 - u)
                if self.isCollision([x3, y3], obstacles_list):
                    dis = self.distance(pose, [x3, y3])
                    data.append(dis + np.random.rand() * self.noise)
                    sense = True
                    break
            if not sense:
                data.append(self.range_max)
        return scan_angles, np.array(data)

# ==============================================================================
# PHẦN 2: SỬ DỤNG API CẤP CAO CỦA PYDECOMP
# ==============================================================================

def main():
    """
    Hàm chính sử dụng API cấp cao của pydecomp.
    """
    # --- STEP 1: SETUP VÀ MÔ PHỎNG LIDAR (Không đổi) ---
    point_A = np.array([0.0, 7.0])
    point_B = np.array([10.0, 7.0])
    uav_pose = np.array([point_A[0], point_A[1], math.pi/4])
    lidar = LidarScanner(range_max=SENSING_RADIUS, angle_min=-math.pi, angle_max=math.pi)

    print("Đang quét môi trường bằng Lidar...")
    scan_angles, scan_ranges = lidar.senseObstacle(uav_pose, OBSTACLES)

    valid_indices = scan_ranges < lidar.range_max
    angles = scan_angles[valid_indices]
    ranges = scan_ranges[valid_indices]
    
    obstacle_points = np.vstack([
        uav_pose[0] + ranges * np.cos(angles),
        uav_pose[1] + ranges * np.sin(angles)
    ]).T

    print(f"Phát hiện {obstacle_points.shape[0]} điểm chướng ngại vật.")
    if obstacle_points.shape[0] == 0:
        print("Không có chướng ngại vật trong tầm quét. Kết thúc.")
        return

    # --- STEP 2: SỬ DỤNG HÀM API MỚI `convex_decomposition_2D` ---
    
    # Tạo đường đi tham chiếu
    # path_reference = np.linspace(point_A, point_B, 20)

    path_reference = np.array([[0, 0], [3, 6], [7, 8]])
    # Tính toán bounding box tự động để cung cấp cho hàm
    all_points = np.vstack([path_reference, obstacle_points])
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    box_size = max_coords - min_coords
    # Thêm một chút padding cho an toàn
    # box = np.array([[box_size[0] + 2, box_size[1] + 2]])
    box = np.array([[2, 2]])
    print("Đang thực hiện phân rã lồi bằng hàm API cấp cao...")
    # Gọi hàm API mới. Nó trả về một danh sách các ma trận A và vector b.
    list_A, list_b = pdc.convex_decomposition_2D(obstacle_points, path_reference, box)

    # --- STEP 3: TRÍCH XUẤT RÀNG BUỘC VÀ TRỰC QUAN HÓA (Dùng hàm có sẵn) ---

    print(f"\nĐã tạo thành công {len(list_A)} đa giác lồi (convex polygon).")
    print("Đây là các ràng buộc tuyến tính (Ax <= b) cho MPC của bạn:")

    for i, (A, b) in enumerate(zip(list_A, list_b)):
        print(f"\n--- Đa giác {i+1} ---")
        print(f"Ma trận A (shape {A.shape}):\n{np.round(A, 3)}")
        print(f"Vector b (shape {b.shape}):\n{np.round(b, 3)}")

    ax = pdc.visualize_environment(Al=list_A, bl=list_b, p=path_reference, planar=True)

    for obs in OBSTACLES:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.6, label='Original Obstacles' if 'Original Obstacles' not in plt.gca().get_legend_handles_labels()[1] else "")
        ax.add_artist(circle)
        
    ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], c='red', s=15, label='Lidar Points', zorder=10)

    ax.set_title('Convex Decomposition using High-Level API')
    ax.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()