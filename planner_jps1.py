"""
jps_planner.py  —  JPS planner TỰ CHỨA (self-contained)

Khác bản cũ:
  - KHÔNG còn import từ planner_astar (OccupancyGrid / _nearest_free /
    project_goal_to_grid / remove_residual_node đều nằm trong file này).
  - Occupancy grid là LOG-ODDS BỀN VỮNG (persistent), trải toàn world,
    tích luỹ quan sát qua các frame -> vật cản không "nhấp nháy".
  - Dùng INVERSE SENSOR MODEL theo tia (raycast):
        * dọc tia từ robot tới bề mặt  -> bằng chứng FREE
        * tại điểm hit                 -> bằng chứng OCCUPIED
        * phía SAU bề mặt + lõi vật cản -> KHÔNG cập nhật -> giữ UNKNOWN
    => JPS không thể plan xuyên lõi vật cản to, vì lõi không bao giờ
       được quan sát là free. Unknown bị coi là CẤM ĐI (bảo thủ) và TỰ
       MỞ KHOÁ khi robot nhìn từ góc khác (thực sự thấy nó trống).
  - Mọi kiểm tra va chạm (planning, smoothing, line-of-sight) đọc CÙNG
    một grid -> hết bug "đường valid nhưng thực ra xuyên vật cản".
  - Vá bug start==goal: trả path 2 điểm (không bị caller đếm là fail).
  - Persistent map SỐNG SÓT dù caller tạo lại JPSPlanner() mỗi lần replan,
    nhờ registry theo agent_id.

Yêu cầu:  pip install pathfind   (giống bản cũ)

------------------------------------------------------------------
TÍCH HỢP Ở CALLER (getOrientedGoalTrajectory) — đổi rất ít:

1) import:
       from jps_planner import JPSPlanner, WORLD_BOUNDS_FROM_SCENARIO

2) Tạo planner MỘT LẦN cho mỗi robot (vd trong __init__ của robot), KHÔNG
   tạo lại mỗi replan. agent_id giúp map bền vững kể cả nếu vẫn lỡ tạo lại:
       self.planner = JPSPlanner(
           world_bounds=(0, -2, 120, 32),   # (xmin, ymin, xmax, ymax) từ scenario xlim/ylim
           agent_id=self.index)

   (Nếu bạn vẫn giữ dòng `self.planner = JPSPlanner()` trong nhánh replan
    thì PHẢI truyền agent_id để map không bị mất:
        self.planner = JPSPlanner(world_bounds=(0,-2,120,32), agent_id=self.index))

3) MỖI control-step (kể cả khi KHÔNG replan), nạp scan LiDAR vào map:
       self.planner.observe(current_robot_pos, obstacle_points)
   -> map càng mượt, shadow co lại đúng lúc robot tiến tới.
   Nếu không gọi được mỗi frame cũng không sao: plan() tự observe.

4) Khi replan, gọi như cũ:
       success, raw_path, _, _, _ = self.planner.plan(
           obstacle_points, ROBOT_RADIUS, time_budget_ms=...)
   Path trả về ĐÃ được smooth sẵn. Dòng
       smoothed = remove_residual_node(raw_path, obstacle_points, ROBOT_RADIUS)
   giờ thừa — có thể bỏ. (remove_residual_node vẫn được export như passthrough
    để không vỡ code cũ.)
------------------------------------------------------------------
"""

import time
import numpy as np

# --- config: lấy hằng nếu có, không thì dùng default ---
try:
    from config import *          # noqa  (ROBOT_RADIUS, SENSING_RADIUS, ...)
except Exception:
    pass

_RES_DEFAULT      = float(globals().get("ASTAR_GRID_RESOLUTION", 0.5))
_ROBOT_R_DEFAULT  = float(globals().get("ROBOT_RADIUS", 0.3))
_INFLATE_DEFAULT  = float(globals().get("ASTAR_INFLATE_RADIUS", _ROBOT_R_DEFAULT + 0.2))
_SENSING_DEFAULT  = float(globals().get("SENSING_RADIUS", 3.0))

try:
    import pathfind
    PATHFIND_AVAILABLE = True
except ImportError:
    PATHFIND_AVAILABLE = False
    print("WARNING: 'pathfind' chưa cài. Chạy: pip install pathfind")


def WORLD_BOUNDS_FROM_SCENARIO(scenario):
    """Tiện ích: lấy (xmin, ymin, xmax, ymax) từ dict scenario có xlim/ylim."""
    x0, x1 = scenario["xlim"]
    y0, y1 = scenario["ylim"]
    return (float(x0), float(y0), float(x1), float(y1))


# ============================================================
# Helpers: shift / dilation nhị phân bằng numpy (không cần scipy)
# ============================================================
def _shift(a, dy, dx):
    """Dịch mảng theo (dy,dx), phần trống = 0 (không wrap-around)."""
    out = np.zeros_like(a)
    H, W = a.shape
    y0, y1 = max(0, -dy), H - max(0, dy)
    x0, x1 = max(0, -dx), W - max(0, dx)
    out[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = a[y0:y1, x0:x1]
    return out


def _dilate_disk(mask, r_cells):
    """Nở (dilation) mask nhị phân bằng đĩa bán kính r_cells."""
    if r_cells <= 0:
        return mask.copy()
    out = mask.copy()
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            if dx * dx + dy * dy <= r_cells * r_cells and (dx or dy):
                out |= _shift(mask, dy, dx)
    return out


# ============================================================
# Persistent log-odds occupancy grid (toàn world, bền vững)
# ============================================================
class PersistentLogOddsGrid:
    """
    Grid log-odds cố định trong khung world (không recenter mỗi frame).
    Trạng thái suy ra từ log-odds L:
        L <= free_logit  -> FREE      (đi được)
        L >= occ_logit   -> OCCUPIED  (vật cản, sẽ được inflate)
        ở giữa / chưa quan sát (L=0) -> UNKNOWN (mặc định CẤM đi)
    """

    def __init__(self, world_bounds, resolution=_RES_DEFAULT,
                 inflate_radius=_INFLATE_DEFAULT, sensing_radius=_SENSING_DEFAULT,
                 n_ray_bins=360,
                 l_occ=0.9, l_free=-0.7, l_min=-2.0, l_max=3.5,
                 occ_thresh=0.65, free_thresh=0.35,
                 treat_unknown_as_blocked=True):
        xmin, ymin, xmax, ymax = world_bounds
        self.ox, self.oy = float(xmin), float(ymin)
        self.resolution = float(resolution)
        self.sensing_radius = float(sensing_radius)
        self.n_bins = int(n_ray_bins)

        self.size_x = int(np.ceil((xmax - xmin) / self.resolution)) + 1
        self.size_y = int(np.ceil((ymax - ymin) / self.resolution)) + 1

        self.L = np.zeros((self.size_y, self.size_x), dtype=np.float32)

        self.l_occ, self.l_free = float(l_occ), float(l_free)
        self.l_min, self.l_max = float(l_min), float(l_max)
        # ngưỡng prob -> logit
        self.occ_logit  = float(np.log(occ_thresh / (1 - occ_thresh)))
        self.free_logit = float(np.log(free_thresh / (1 - free_thresh)))

        self.treat_unknown_as_blocked = bool(treat_unknown_as_blocked)
        self.set_inflate(inflate_radius)

        self.blocked = np.zeros_like(self.L, dtype=bool)
        self._dirty = True

    # ---- inflate ----
    def set_inflate(self, radius):
        self.inflate_radius = float(radius)
        self.inflate_cells = int(np.ceil(self.inflate_radius / self.resolution))

    # ---- toạ độ ----
    def world_to_grid(self, wx, wy):
        ix = int((wx - self.ox) / self.resolution)
        iy = int((wy - self.oy) / self.resolution)
        return ix, iy

    def grid_to_world(self, ix, iy):
        wx = self.ox + (ix + 0.5) * self.resolution
        wy = self.oy + (iy + 0.5) * self.resolution
        return wx, wy

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.size_x and 0 <= iy < self.size_y

    def is_within_world(self, wx, wy):
        return (self.ox <= wx < self.ox + self.size_x * self.resolution and
                self.oy <= wy < self.oy + self.size_y * self.resolution)

    # ---- nạp linear index hợp lệ từ mảng toạ độ world ----
    def _lin(self, xs, ys):
        ix = ((xs - self.ox) / self.resolution).astype(np.int64)
        iy = ((ys - self.oy) / self.resolution).astype(np.int64)
        m = (ix >= 0) & (ix < self.size_x) & (iy >= 0) & (iy < self.size_y)
        return iy[m] * self.size_x + ix[m]

    # ==========================================================
    # OBSERVE: cập nhật map từ 1 lần quét LiDAR
    #   robot_pos  : (x, y) gốc tia
    #   hit_points : Nx2(3) điểm LiDAR đập vào bề mặt vật cản
    # ==========================================================
    def observe(self, robot_pos, hit_points):
        rx, ry = float(robot_pos[0]), float(robot_pos[1])
        res = self.resolution
        step = res * 0.5

        # gom min-range theo bin góc
        bin_range = np.full(self.n_bins, np.inf, dtype=np.float64)
        hp = None
        if hit_points is not None and len(hit_points) > 0:
            hp = np.asarray(hit_points, dtype=float)[:, :2]
            d = hp - np.array([rx, ry])
            rng = np.hypot(d[:, 0], d[:, 1])
            ang = np.arctan2(d[:, 1], d[:, 0])               # [-pi, pi]
            b = ((ang + np.pi) / (2 * np.pi) * self.n_bins).astype(int) % self.n_bins
            for bi, ri in zip(b, rng):
                if ri < bin_range[bi]:
                    bin_range[bi] = ri

        free_idx = []
        occ_idx = []

        two_pi = 2 * np.pi
        for bi in range(self.n_bins):
            ang = -np.pi + (bi + 0.5) * (two_pi / self.n_bins)
            ca, sa = np.cos(ang), np.sin(ang)
            r_hit = bin_range[bi]
            seen = np.isfinite(r_hit) and r_hit <= self.sensing_radius
            free_len = min(r_hit, self.sensing_radius) if np.isfinite(r_hit) \
                else self.sensing_radius

            # FREE dọc tia tới ngay trước bề mặt
            n = int(max((free_len - res) / step, 0))
            if n > 0:
                rr = np.arange(n) * step
                xs = rx + rr * ca
                ys = ry + rr * sa
                lin = self._lin(xs, ys)
                if lin.size:
                    free_idx.append(np.unique(lin))   # unique trong cùng tia

            # OCCUPIED tại điểm hit
            if seen:
                hx, hy = rx + r_hit * ca, ry + r_hit * sa
                lin = self._lin(np.array([hx]), np.array([hy]))
                if lin.size:
                    occ_idx.append(lin)

        # mark thẳng các điểm hit thật (giữ độ nét bề mặt)
        if hp is not None:
            lin = self._lin(hp[:, 0], hp[:, 1])
            if lin.size:
                occ_idx.append(lin)

        Lflat = self.L.reshape(-1)
        if free_idx:
            np.add.at(Lflat, np.concatenate(free_idx), self.l_free)
        if occ_idx:
            np.add.at(Lflat, np.concatenate(occ_idx), self.l_occ)

        np.clip(self.L, self.l_min, self.l_max, out=self.L)
        self._dirty = True

    # ==========================================================
    # REBUILD: dựng lớp planning (blocked) từ log-odds
    # ==========================================================
    def rebuild(self):
        occ = self.L >= self.occ_logit
        blocked = _dilate_disk(occ, self.inflate_cells)
        if self.treat_unknown_as_blocked:
            free = self.L <= self.free_logit
            unknown = (~free) & (~occ)
            blocked |= unknown
        self.blocked = blocked
        self._dirty = False

    def _ensure(self):
        if self._dirty:
            self.rebuild()

    # ---- truy vấn cho planner ----
    def is_blocked(self, ix, iy):
        if not self.in_bounds(ix, iy):
            return True
        self._ensure()
        return bool(self.blocked[iy, ix])

    def is_free(self, ix, iy):
        return self.in_bounds(ix, iy) and not self.is_blocked(ix, iy)

    def to_matrix(self):
        """matrix cho pathfind: -1 = obstacle, 1 = walkable. row=iy, col=ix."""
        self._ensure()
        return np.where(self.blocked, -1, 1).tolist()

    # ---- nearest free (xoáy ốc) ----
    def nearest_free(self, ix, iy, max_radius=20):
        self._ensure()
        if self.is_free(ix, iy):
            return ix, iy
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = ix + dx, iy + dy
                    if self.is_free(nx, ny):
                        return nx, ny
        return None, None

    # ---- line-of-sight: đoạn p0->p1 có sạch không (cùng grid!) ----
    def segment_clear(self, wp0, wp1):
        self._ensure()
        p0 = np.asarray(wp0[:2], float)
        p1 = np.asarray(wp1[:2], float)
        d = p1 - p0
        dist = float(np.hypot(d[0], d[1]))
        if dist < 1e-9:
            ix, iy = self.world_to_grid(p0[0], p0[1])
            return self.is_free(ix, iy)
        n = int(dist / (self.resolution * 0.5)) + 1
        ts = np.linspace(0.0, 1.0, n)
        xs = p0[0] + ts * d[0]
        ys = p0[1] + ts * d[1]
        for wx, wy in zip(xs, ys):
            ix, iy = self.world_to_grid(wx, wy)
            if self.is_blocked(ix, iy):
                return False
        return True


# ============================================================
# Goal projection (rút gọn): kẹp goal vào world nếu nằm ngoài
# ============================================================
def project_goal_to_grid(start, goal, grid: PersistentLogOddsGrid, margin=2.0):
    start = np.asarray(start[:2], float)
    goal = np.asarray(goal[:2], float)
    if grid.is_within_world(goal[0], goal[1]):
        return goal, False
    xmin = grid.ox + margin
    ymin = grid.oy + margin
    xmax = grid.ox + grid.size_x * grid.resolution - margin
    ymax = grid.oy + grid.size_y * grid.resolution - margin
    clamped = np.array([np.clip(goal[0], xmin, xmax),
                        np.clip(goal[1], ymin, ymax)])
    return clamped, True


# ============================================================
# Smoothing: string-pulling dùng line-of-sight trên CÙNG grid
# ============================================================
def _string_pull(path_world, grid: PersistentLogOddsGrid):
    if len(path_world) <= 2:
        return [list(p) for p in path_world]
    pts = [np.asarray(p[:2], float) for p in path_world]
    out = [pts[0]]
    anchor = 0
    i = 1
    while i < len(pts):
        # cố nối anchor -> i+? xa nhất còn line-of-sight
        if i == len(pts) - 1 or not grid.segment_clear(pts[anchor], pts[i + 1]):
            if not grid.segment_clear(pts[anchor], pts[i]):
                # đoạn anchor->i cũng kẹt: giữ điểm i-1 làm mốc an toàn
                out.append(pts[i - 1])
                anchor = i - 1
            else:
                out.append(pts[i])
                anchor = i
        i += 1
    if not np.allclose(out[-1], pts[-1]):
        out.append(pts[-1])
    return [list(p) for p in out]


def remove_residual_node(path, obstacle_points=None, robot_radius=None):
    """Passthrough để tương thích code cũ (smoothing đã làm trong plan())."""
    return np.asarray(path, dtype=float)


# ============================================================
# JPS planner (drop-in)
# ============================================================
class JPSPlanner:

    # registry: map bền vững theo agent_id, sống sót dù planner bị tạo lại
    _MAP_REGISTRY = {}

    def __init__(self, world_bounds=None, agent_id=None,
                 grid_resolution=_RES_DEFAULT,
                 inflate_radius=_INFLATE_DEFAULT,
                 sensing_radius=_SENSING_DEFAULT,
                 treat_unknown_as_blocked=True,
                 **grid_kwargs):
        if not PATHFIND_AVAILABLE:
            raise ImportError("Library 'pathfind' chưa cài. Chạy: pip install pathfind")
        if world_bounds is None:
            raise ValueError(
                "JPSPlanner cần world_bounds=(xmin,ymin,xmax,ymax). "
                "Dùng WORLD_BOUNDS_FROM_SCENARIO(scenario) để lấy từ xlim/ylim.")

        self.agent_id = agent_id
        self.start = None
        self.goal = None

        key = (agent_id, tuple(np.round(world_bounds, 3)),
               round(grid_resolution, 4))
        if agent_id is not None and key in JPSPlanner._MAP_REGISTRY:
            self.gmap = JPSPlanner._MAP_REGISTRY[key]
            self.gmap.set_inflate(inflate_radius)
        else:
            self.gmap = PersistentLogOddsGrid(
                world_bounds, resolution=grid_resolution,
                inflate_radius=inflate_radius, sensing_radius=sensing_radius,
                treat_unknown_as_blocked=treat_unknown_as_blocked, **grid_kwargs)
            if agent_id is not None:
                JPSPlanner._MAP_REGISTRY[key] = self.gmap

    # ---- API ----
    def initialize(self, start, goal):
        self.start = np.asarray(start[:2], dtype=float)
        self.goal = np.asarray(goal[:2], dtype=float)

    def observe(self, robot_pos, obstacle_points):
        """Gọi MỖI control-step để map luôn tươi (kể cả khi không replan)."""
        self.gmap.observe(robot_pos, obstacle_points)

    @classmethod
    def reset_map(cls, agent_id=None):
        """Xoá map (vd khi bắt đầu episode mới)."""
        if agent_id is None:
            cls._MAP_REGISTRY.clear()
        else:
            for k in [k for k in cls._MAP_REGISTRY if k[0] == agent_id]:
                del cls._MAP_REGISTRY[k]

    def plan(self, obstacle_points, robot_radius,
             max_iter=None, time_budget_ms=None, observe=True):
        if self.start is None or self.goal is None:
            return False, [], set(), set(), 0

        # 1) cập nhật map từ scan hiện tại
        if observe:
            self.gmap.observe(self.start, obstacle_points)
        self.gmap.set_inflate(max(self.gmap.inflate_radius, robot_radius))
        self.gmap.rebuild()

        # 2) toạ độ grid
        sx, sy = self.gmap.world_to_grid(self.start[0], self.start[1])
        local_goal, _ = project_goal_to_grid(self.start, self.goal, self.gmap)
        gx, gy = self.gmap.world_to_grid(local_goal[0], local_goal[1])

        if not self.gmap.in_bounds(sx, sy) or not self.gmap.in_bounds(gx, gy):
            return False, [], set(), set(), 0

        # 3) start kẹt -> nhích ra ô free gần nhất
        if not self.gmap.is_free(sx, sy):
            r = self.gmap.nearest_free(sx, sy, max_radius=20)
            if r[0] is None:
                return False, [], set(), set(), 0
            sx, sy = r

        # 4) goal kẹt (trong vật cản HOẶC trong vùng unknown/shadow)
        #    -> kéo về ô free gần goal nhất = điểm biên (frontier) hướng goal.
        #    bán kính lớn để vượt qua bóng/khối vật cản to.
        if not self.gmap.is_free(gx, gy):
            r = self.gmap.nearest_free(gx, gy, max_radius=60)
            if r[0] is None:
                return False, [], set(), set(), 0
            gx, gy = r

        # 5) start == goal: đã ở/đạt vị trí khả thi gần nhất.
        #    Trả PATH 2 ĐIỂM (không phải 1) để caller KHÔNG đếm là fail.
        if (sx, sy) == (gx, gy):
            wx, wy = self.gmap.grid_to_world(sx, sy)
            return True, [[wx, wy], [wx, wy]], set(), set(), 2

        # 6) JPS qua pathfind
        matrix = self.gmap.to_matrix()
        start_str = f"{sy},{sx}"     # pathfind: "row,col" = (y,x)
        end_str = f"{gy},{gx}"
        try:
            graph = pathfind.transform.matrix2graph(matrix, diagonal=True)
            path_strs = pathfind.find(graph, start=start_str, end=end_str,
                                      method="jps")
        except Exception as e:
            print(f"JPS library error: {type(e).__name__}: {e}")
            return False, [], set(), set(), 0

        if not path_strs:
            return False, [], set(), set(), 0

        # 7) về world coords
        path_world = []
        for s in path_strs:
            iy, ix = (int(v) for v in s.split(","))
            wx, wy = self.gmap.grid_to_world(ix, iy)
            path_world.append([wx, wy])

        # 8) smoothing (line-of-sight trên CÙNG grid)
        smoothed = _string_pull(path_world, self.gmap)
        if len(smoothed) < 2:
            smoothed = path_world if len(path_world) >= 2 else \
                [path_world[0], path_world[0]]

        return True, smoothed, set(), set(), len(smoothed)


# ============================================================
# Smoke test (numpy-only, không cần pathfind)
# ============================================================
if __name__ == "__main__":
    g = PersistentLogOddsGrid((0, -2, 120, 32), resolution=0.5,
                              inflate_radius=1.0, sensing_radius=60.0)
    robot = (10.0, 15.0)
    # giả lập LiDAR thấy bề mặt TRƯỚC của 1 rect [32,20,6,8] (x in 32..38, y 20..28)
    # bề mặt trái (x=32) hướng về robot:
    surf = np.array([[32.0, y] for y in np.linspace(20, 28, 40)])
    g.observe(robot, surf)
    g.rebuild()

    # ô ngay trước bề mặt -> free; lõi (x=35,y=24) -> KHÔNG free (unknown=blocked)
    ix, iy = g.world_to_grid(25, 24)
    core_ix, core_iy = g.world_to_grid(35, 24)
    print("trước bề mặt free? ", g.is_free(ix, iy))        # mong: True
    print("lõi vật cản free? ", g.is_free(core_ix, core_iy))  # mong: False
    print("line-of-sight xuyên lõi sạch? ",
          g.segment_clear((25, 24), (45, 24)))             # mong: False
    print("grid size:", g.size_x, "x", g.size_y)