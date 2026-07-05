"""
  (1) Execution time  — mean
  (2) Total path length
  (3) FOV coverage area
      + coverage efficiency = area / (N * (2L)^2)  ∈ [0, 1]
  (4) Tracking:
      + visibility_ratio (%)
      + track_err (m)
"""

import numpy as np


# ────────────────────────────────────────────────────────────
# (A) Union diện tích các hình chữ nhật trục-chuẩn (exact)
#     Thuật toán coordinate-compression, O(n^2) — đủ cho n<=~10 UAV.
# ────────────────────────────────────────────────────────────
def rect_union_area(rects):
    """
    rects: list các (x0, y0, x1, y1). Trả về diện tích hợp (union), exact.
    Phần chồng lấn giữa các rect chỉ được đếm MỘT lần.
    """
    rects = [r for r in rects if r[2] > r[0] and r[3] > r[1]]
    if not rects:
        return 0.0
    xs = sorted(set([r[0] for r in rects] + [r[2] for r in rects]))
    total = 0.0
    for i in range(len(xs) - 1):
        x_lo, x_hi = xs[i], xs[i + 1]
        w = x_hi - x_lo
        if w <= 0:
            continue
        # các rect phủ trọn dải x này -> gom khoảng y
        y_iv = []
        for (rx0, ry0, rx1, ry1) in rects:
            if rx0 <= x_lo and rx1 >= x_hi:
                y_iv.append((ry0, ry1))
        if not y_iv:
            continue
        y_iv.sort()
        cov = 0.0
        cur_lo, cur_hi = y_iv[0]
        for lo, hi in y_iv[1:]:
            if lo > cur_hi:            # rời nhau  -> chốt khoảng cũ
                cov += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
            else:                      # chồng/chạm -> nuốt vào (khử phần lấn)
                cur_hi = max(cur_hi, hi)
        cov += cur_hi - cur_lo
        total += w * cov
    return float(total)


def _fov_rects(centers, side):
    """centers (N,2) -> list rect (x0,y0,x1,y1) của FOV vuông cạnh `side`."""
    h = side / 2.0
    return [(c[0] - h, c[1] - h, c[0] + h, c[1] + h) for c in centers]


def _connected_components(centers, side, connect_slack):
    """
    Đồ thị liên thông giữa FOV: nối nếu 2 hình vuông chồng/chạm nhau
    (cho phép hở tối đa `connect_slack`). Trả về list các list index.

    Với side = 2*VIEWING_RADIUS và connect_slack = 0 (mặc định, strict):
    hai FOV nối nhau  <=>  dx <= 2R VÀ dy <= 2R  (chạm/chồng đúng nghĩa).
    Chỉ cần hở ra ở bất kỳ trục nào (> 2R) là coi như RỜI.
    """
    n = len(centers)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i][0] - centers[j][0])
            dy = abs(centers[i][1] - centers[j][1])
            if dx <= side + connect_slack and dy <= side + connect_slack:
                union(i, j)

    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def fov_coverage(centers, side, target=None, leader_idx=None,
                 connect_slack=None, mode="leader_component"):
    """
    Diện tích hợp của các FOV vuông cạnh `side`, có LỌC FOV tách rời.

    Luật cơ bản:
      - RỜI nhau  -> KHÔNG tính vào tổng diện tích.
      - CHỒNG lấn -> phần lấn chỉ đếm MỘT lần (union).

    mode:
      "leader_component" : chỉ giữ component chứa leader (cần leader_idx).
                           UAV nào lạc khỏi leader — không liên thông về
                           leader qua chuỗi FOV chồng nhau — đều bị bỏ.
      "drop_isolated"    : loại các FOV thuộc component cỡ 1 (cô lập).
      "target_component" : chỉ giữ component chứa `target` (cần target).
      "largest"          : chỉ giữ component lớn nhất.

    connect_slack: khe hở tối đa để 2 FOV vẫn coi là liên thông.
                   MẶC ĐỊNH = 0 (strict): chỉ cần hở ra ở bất kỳ trục nào
                   (dx hoặc dy > 2*VIEWING_RADIUS = side) là coi như rời.

    Trả về (area, kept_indices).
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    if n == 0:
        return 0.0, []
    if connect_slack is None:
        connect_slack = 0

    comps = _connected_components(centers, side, connect_slack)

    if mode == "leader_component":
        assert leader_idx is not None
        kept = next((list(comp) for comp in comps if leader_idx in comp), [])
    elif mode == "drop_isolated":
        kept = [i for comp in comps if len(comp) >= 2 for i in comp]
    elif mode == "largest":
        big = max(comps, key=len)
        kept = list(big)
    elif mode == "target_component":
        assert target is not None
        t = np.asarray(target[:2], dtype=float)
        h = side / 2.0
        kept = []
        for comp in comps:
            for i in comp:
                if (abs(centers[i][0] - t[0]) <= h and
                        abs(centers[i][1] - t[1]) <= h):
                    kept = list(comp)
                    break
            if kept:
                break
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")

    if not kept:
        return 0.0, []
    rects = _fov_rects(centers[kept], side)
    return rect_union_area(rects), kept


# ────────────────────────────────────────────────────────────
# (B) Metrics cho 1 run (đọc từ data pickle của main.py)
# ────────────────────────────────────────────────────────────
def run_metrics(data, fov_side, robot_radius, timestep,
                cov_mode="leader_component", connect_slack=None,
                warmup_steps=0):
    """
    data: dict pickle từ main.py (keys: int robot idx + "meta").
    fov_side = 2 * VIEWING_RADIUS.
    warmup_steps: bỏ qua N bước đầu (giai đoạn tiếp cận) khi tính coverage/vis.

    Leader mỗi frame = UAV gần target nhất (argmin khoảng cách Euclid).

    Trả về dict scalar cho 1 run.
    """
    L = fov_side / 2.0
    keys = sorted(k for k in data.keys() if isinstance(k, int))
    n = len(keys)

    paths = [np.asarray(data[i]["path"]) for i in keys]
    T = min(p.shape[0] for p in paths)
    pos = np.stack([p[:T, 1:3] for p in paths], axis=1)    # (T, n, 2)
    ctrl = np.stack([p[:T, 7:10] for p in paths], axis=1)  # (T, n, 3)

    tar = np.asarray(data[keys[0]]["tar_traj"])[:T, :2]    # (T, 2)

    s = min(warmup_steps, T - 1)

    # ── khoảng cách UAV -> target (dùng cho cả leader và tracking) ──
    diff = pos - tar[:, None, :]                # (T, n, 2)
    d_inf = np.max(np.abs(diff), axis=2)        # Chebyshev (T, n)
    d_euc = np.linalg.norm(diff, axis=2)        # (T, n)
    leader_seq = d_euc.argmin(axis=1)           # (T,) UAV gần target nhất mỗi frame

    # ── (3) FOV coverage theo thời gian ──
    cov_series = np.zeros(T)
    for t in range(T):
        area, _ = fov_coverage(pos[t], fov_side,
                               leader_idx=int(leader_seq[t]),
                               connect_slack=connect_slack, mode=cov_mode)
        cov_series[t] = area
    ideal = n * (fov_side ** 2)
    cov_mean = float(cov_series[s:].mean())
    cov_eff = float((cov_series[s:] / ideal).mean())

    # ── (4) Tracking ── (diff/d_inf/d_euc đã tính ở trên)
    visible = d_inf.min(axis=1) <= L                 # target trong >=1 FOV
    vis_ratio = 100.0 * visible[s:].mean()
    track_err = float(d_euc.min(axis=1)[s:].mean())  # tới UAV gần nhất

    # ── (2) Path length ──
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=2)  # (T-1, n)
    path_len_per = seg.sum(axis=0)
    total_path_len = float(path_len_per.sum())
    mean_path_len = float(path_len_per.mean())

    # ── control energy (bonus) ──
    energy = float((np.linalg.norm(ctrl, axis=2) ** 2).sum() * timestep)

    # ── (1) Execution time — CHỈ lấy giá trị trung bình đã lưu ──
    meta = data.get("meta", {})
    exec_mean = meta.get("compute_time_mean", None)
    if exec_mean is None:                    # fallback nếu chỉ lưu mảng
        ct = np.asarray(meta.get("compute_times", []))
        exec_mean = float(ct.mean()) if ct.size else np.nan
    exec_mean = float(exec_mean)

    # ── safety (bonus) ──
    min_pair = np.inf
    for a in range(n):
        for b in range(a + 1, n):
            dmin = np.linalg.norm(pos[:, a] - pos[:, b], axis=1).min()
            min_pair = min(min_pair, dmin)
    min_inter_uav = float(min_pair) if n > 1 else np.nan

    return dict(
        exec_time=exec_mean,
        total_path_len=total_path_len,
        mean_path_len=mean_path_len,
        fov_area=cov_mean,
        fov_coverage_eff=cov_eff,
        visibility=vis_ratio,
        track_err=track_err,
        control_energy=energy,
        min_inter_uav=min_inter_uav,
        target_reached=meta.get("target_reached", None),
        sim_time=float(paths[0][T - 1, 0]),
    )