"""
formation.py — logic đội hình DÙNG CHUNG cho cả 3 phương pháp
=============================================================
Mục tiêu: để so sánh CÔNG BẰNG, cả APF / Pure MPC / (proposed) đều nhắm tới
CÙNG một đội hình (cùng leader, cùng vị trí slot). Khác biệt duy nhất còn lại
là TẦNG TRÁNH VẬT CẢN / PLANNING của mỗi phương pháp.

Hình học lấy khớp với robot_jps1.costSatelliteSlotDynamic:
    - Leader = UAV gần target nhất theo Chebyshev, sticky khi còn trong FOV.
    - side = SLOT_SPACING_FACTOR * VIEWING_RADIUS  (mặc định 4.0, KHỚP code hiện tại)
    - slot_pos = leader_pos + cell * side,  cell ∈ GRID_CELLS
    - Gán Hungarian (satellite × slot) + hysteresis SWITCH_MARGIN.
    - SEARCH (không ai thấy target): trả về goal = target cho MỌI UAV.

⚠ side=4L khiến các FOV RỜI NHAU. Nếu muốn FOV tile khít, đặt
   SLOT_SPACING_FACTOR = 2.0 (cả ở đây và trong robot_jps1 để nhất quán).

Cách dùng trong robot (APF / Pure MPC):
    role, goal_xy = get_formation_goal(self, robots)
    # role ∈ {"search", "leader", "satellite"};  goal_xy = np.array(2,)
"""

import numpy as np
from config import (VIEWING_RADIUS, GRID_CELLS, SWITCH_MARGIN,
                    OPEN_ALL_SLOTS, VISIBILITY_MARGIN_RATIO)

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

SLOT_SPACING_FACTOR = 2


# ────────────────────────────────────────────────────────────
def _select_leader(robots, target_xy, prev_leader):
    """Chọn leader: sticky nếu leader cũ còn thấy target, else UAV gần nhất."""
    L = VIEWING_RADIUS
    L_strict = L * (1 - VISIBILITY_MARGIN_RATIO)

    d_inf = {}
    seers_strict, seers_relaxed = [], []
    for r in robots:
        d = max(abs(r.state[0] - target_xy[0]), abs(r.state[1] - target_xy[1]))
        d_inf[r.index] = d
        if d <= L_strict:
            seers_strict.append(r.index)
        if d <= L:
            seers_relaxed.append(r.index)

    if prev_leader is not None and prev_leader in seers_relaxed:
        return prev_leader, d_inf
    if seers_strict:
        return min(seers_strict, key=lambda i: d_inf[i]), d_inf
    if seers_relaxed:
        return min(seers_relaxed, key=lambda i: d_inf[i]), d_inf
    return None, d_inf


def _assign_slots(sat_indices, pos_of, leader_pos, prev_assign):
    """Hungarian satellite×slot + hysteresis. Trả về dict index->slot_cell_idx."""
    sats = sorted(sat_indices)
    n = len(sats)
    if n == 0:
        return {}

    side = SLOT_SPACING_FACTOR * VIEWING_RADIUS
    if OPEN_ALL_SLOTS:
        m = min(4, len(GRID_CELLS))
    else:
        m = min(n, len(GRID_CELLS))
    cells = GRID_CELLS[:m]
    slot_pos = [np.asarray(leader_pos) + np.array([dx, dy]) * side
                for dx, dy in cells]

    C = np.array([[float(np.linalg.norm(pos_of[i] - slot_pos[j]))
                   for j in range(m)] for i in sats])

    if _HAS_SCIPY:
        row, col = linear_sum_assignment(C)
        new_assign = {sats[r]: int(c) for r, c in zip(row, col)}
    else:
        new_assign, used = {}, set()
        order = sorted((C[a][b], a, b) for a in range(n) for b in range(m))
        for _, a, b in order:
            if sats[a] in new_assign or b in used:
                continue
            new_assign[sats[a]] = b
            used.add(b)
    new_cost = sum(C[sats.index(i)][new_assign[i]] for i in sats)

    # Hysteresis: giữ assignment cũ nếu mới không rẻ hơn SWITCH_MARGIN
    if (prev_assign is not None and all(i in prev_assign for i in sats)
            and all(prev_assign[i] < m for i in sats)):
        prev_cost = sum(C[sats.index(i)][prev_assign[i]] for i in sats)
        if new_cost > prev_cost - SWITCH_MARGIN:
            new_assign = prev_assign
    return new_assign


def get_formation_goal(robot, robots):
    """
    Tính (role, goal_xy) cho `robot` trong cycle hiện tại.

    role:
      "search"    -> chưa ai thấy target: goal = target
      "leader"    -> robot là leader: goal = target (target ở tâm FOV)
      "satellite" -> goal = vị trí slot được gán

    Trạng thái hysteresis lưu trên robot: robot._fm_leader, robot._fm_assign.
    """
    target_xy = np.asarray(robot.goal[:2], dtype=float)

    prev_leader = getattr(robot, "_fm_leader", None)
    leader_idx, _ = _select_leader(robots, target_xy, prev_leader)
    robot._fm_leader = leader_idx

    # SEARCH: chưa ai thấy target -> mọi UAV bám target
    if leader_idx is None:
        return "search", target_xy

    # self là leader
    if robot.index == leader_idx:
        return "leader", target_xy

    # self là satellite -> cần vị trí leader + gán slot
    leader_pos = None
    for r in robots:
        if r.index == leader_idx:
            leader_pos = np.asarray(r.state[:2], dtype=float)
            break
    if leader_pos is None:
        return "search", target_xy

    sat_indices = [r.index for r in robots if r.index != leader_idx]
    pos_of = {r.index: np.asarray(r.state[:2], dtype=float)
              for r in robots if r.index in sat_indices}

    prev_assign = getattr(robot, "_fm_assign", None)
    assign = _assign_slots(sat_indices, pos_of, leader_pos, prev_assign)
    robot._fm_assign = assign

    if robot.index not in assign:
        return "search", target_xy

    side = SLOT_SPACING_FACTOR * VIEWING_RADIUS
    dx, dy = GRID_CELLS[assign[robot.index]]
    slot_xy = leader_pos + np.array([dx, dy]) * side
    return "satellite", slot_xy