"""
Cache module cho target trajectory.

Lý do tồn tại:
- SequentialRRTPlanner dùng RRT random, mỗi lần chạy main cho trajectory khác.
- Khi debug/tune MPC của UAV, ta muốn target trajectory CỐ ĐỊNH để so sánh
  giữa các lần chạy.
- Đồng thời tránh tốn thời gian gen lại nếu waypoints không đổi.

Cách dùng:
    from target_cache import load_or_generate_target

    target = load_or_generate_target(TAR_WAYPOINTS)
    # target là SequentialRRTPlanner đã sẵn sàng (.state, .trajectory, ...)

Force regenerate:
    target = load_or_generate_target(TAR_WAYPOINTS, force_regen=True)
"""

import os
import pickle
import hashlib
import time
import numpy as np

from config import (SCENARIO, TAR_MAX_SPEED, TIMESTEP,
                    TAR_STEP_LENGTH, TAR_GOAL_SAMPLE_RATE,
                    TAR_MAX_ITER, SAFETY_MARGIN,
                    TAR_SMOOTH_ENABLE, TAR_SMOOTH_METHOD,
                    TAR_SMOOTH_ITERATIONS, TAR_SPLINE_DS)
from target_rrt import SequentialRRTPlanner


CACHE_DIR = "cache_target"


# ============================================================
# Cache key — đổi config thì hash đổi -> tự gen lại
# ============================================================
def _make_cache_key(waypoints):
    """Hash của (waypoints + các tham số ảnh hưởng tới trajectory)."""
    h = hashlib.md5()
    for wp in waypoints:
        h.update(np.asarray(wp, dtype=np.float64).tobytes())
    # Tham số ảnh hưởng tới trajectory (gồm cả smoothing)
    params = (SCENARIO, TAR_MAX_SPEED, TIMESTEP,
              TAR_STEP_LENGTH, TAR_GOAL_SAMPLE_RATE,
              TAR_MAX_ITER, SAFETY_MARGIN,
              TAR_SMOOTH_ENABLE, TAR_SMOOTH_METHOD,
              TAR_SMOOTH_ITERATIONS, TAR_SPLINE_DS)
    h.update(str(params).encode())
    return h.hexdigest()[:12]   # 12 ký tự đủ tránh collision


def _cache_path(waypoints):
    key = _make_cache_key(waypoints)
    return os.path.join(CACHE_DIR, f"target_scen{SCENARIO}_{key}.pkl")


# ============================================================
# Save / Load
# ============================================================
def _save_target_cache(target, waypoints, cache_file):
    """
    Lưu trajectory + metadata. KHÔNG pickle nguyên object vì:
    - Bên trong có thể có reference tới RRT, Node, ... khó deserialize ổn định
    - Chỉ cần state + trajectory là đủ để recreate
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    data = {
        'version': 1,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scenario': SCENARIO,
        'waypoints': [np.asarray(wp) for wp in waypoints],
        'trajectory': [np.asarray(p) for p in target.trajectory],
        'final_destination': np.asarray(target.final_destination),
        'params': {
            'TAR_MAX_SPEED': TAR_MAX_SPEED,
            'TIMESTEP': TIMESTEP,
            'TAR_STEP_LENGTH': TAR_STEP_LENGTH,
            'TAR_GOAL_SAMPLE_RATE': TAR_GOAL_SAMPLE_RATE,
            'TAR_MAX_ITER': TAR_MAX_ITER,
            'SAFETY_MARGIN': SAFETY_MARGIN,
        },
        'n_points': len(target.trajectory),
        'total_distance': _compute_total_distance(target.trajectory),
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_target_cache(cache_file, waypoints):
    """
    Load cache + reconstruct SequentialRRTPlanner.
    Trả về (target, metadata) hoặc (None, reason) nếu invalid.
    """
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
        return None, f"failed to read: {e}"

    # Validate version
    if data.get('version') != 1:
        return None, f"version mismatch: {data.get('version')}"

    # Validate waypoints (dù đã hash, vẫn check kỹ để chắc chắn)
    cached_wps = data['waypoints']
    if len(cached_wps) != len(waypoints):
        return None, "waypoints count mismatch"
    for a, b in zip(cached_wps, waypoints):
        if not np.allclose(np.asarray(a), np.asarray(b), atol=1e-6):
            return None, "waypoints values mismatch"

    # Validate scenario
    if data.get('scenario') != SCENARIO:
        return None, f"scenario mismatch: cached={data.get('scenario')}, "\
                     f"current={SCENARIO}"

    # Reconstruct SequentialRRTPlanner KHÔNG chạy generateTrajectory()
    target = SequentialRRTPlanner.__new__(SequentialRRTPlanner)
    # set các attribute cần thiết (theo target_rrt.py)
    target.waypoints_3d = [np.array(wp) for wp in waypoints]
    target.waypoints_2d = [tuple(wp[:2]) for wp in target.waypoints_3d]
    target.trajectory = list(data['trajectory'])
    target.state = target.trajectory[0].copy()
    target.final_destination = data['final_destination']
    target.traj_index = 0
    # Các obstacle config — set lại từ config hiện tại (có thể đổi)
    from config import RECTANGLE_OBSTACLES, OBSTACLES, TAR_RADIUS
    target.obs_rect = RECTANGLE_OBSTACLES
    target.obs_circ = OBSTACLES
    target.clearance_radius = TAR_RADIUS + SAFETY_MARGIN

    return target, data


def _compute_total_distance(trajectory):
    """Tổng quãng đường để đo metadata."""
    total = 0.0
    for i in range(1, len(trajectory)):
        total += float(np.linalg.norm(
            np.asarray(trajectory[i]) - np.asarray(trajectory[i - 1])))
    return total


# ============================================================
# Public API
# ============================================================
def load_or_generate_target(waypoints, force_regen=False, verbose=True):
    """
    Load target trajectory từ cache nếu có, không thì generate mới và save.

    Args:
        waypoints: list các waypoint [x, y, z] (giống TAR_WAYPOINTS)
        force_regen: True thì luôn gen lại, ghi đè cache
        verbose: in log

    Returns:
        target: SequentialRRTPlanner đã sẵn sàng (.state, .trajectory, ...)
    """
    cache_file = _cache_path(waypoints)

    # Thử load
    if not force_regen and os.path.exists(cache_file):
        target, meta = _load_target_cache(cache_file, waypoints)
        if target is not None:
            if verbose:
                print(f"[CACHE] Loaded target from {cache_file}")
                print(f"        created_at  : {meta['created_at']}")
                print(f"        n_points    : {meta['n_points']}")
                print(f"        total_dist  : {meta['total_distance']:.2f}")
            return target
        else:
            if verbose:
                print(f"[CACHE] Invalid cache ({meta}), regenerating...")

    # Generate mới
    if verbose:
        print(f"[CACHE] No valid cache, generating target trajectory...")
    t0 = time.time()
    target = SequentialRRTPlanner(waypoints)
    target.generateTrajectory()
    gen_time = time.time() - t0
    if verbose:
        print(f"[CACHE] Generated in {gen_time:.2f}s "
              f"({len(target.trajectory)} points)")

    # Lưu
    _save_target_cache(target, waypoints, cache_file)
    if verbose:
        print(f"[CACHE] Saved to {cache_file}")

    return target


def clear_cache():
    """Xóa toàn bộ cache. Tiện khi muốn reset hoàn toàn."""
    if not os.path.exists(CACHE_DIR):
        return
    for f in os.listdir(CACHE_DIR):
        if f.startswith('target_') and f.endswith('.pkl'):
            os.remove(os.path.join(CACHE_DIR, f))
            print(f"[CACHE] Removed {f}")


def list_cache():
    """In ra các file cache hiện có."""
    if not os.path.exists(CACHE_DIR):
        print("[CACHE] No cache directory")
        return
    files = [f for f in os.listdir(CACHE_DIR)
             if f.startswith('target_') and f.endswith('.pkl')]
    if not files:
        print("[CACHE] No cache files")
        return
    print(f"[CACHE] {len(files)} cache file(s):")
    for f in files:
        path = os.path.join(CACHE_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    from config import TAR_WAYPOINTS

    print("=" * 60)
    print("Test 1: First call (sẽ generate)")
    print("=" * 60)
    target1 = load_or_generate_target(TAR_WAYPOINTS)
    print(f"State[0]: {target1.state}")
    print(f"Trajectory length: {len(target1.trajectory)}")

    print("\n" + "=" * 60)
    print("Test 2: Second call (sẽ load cache)")
    print("=" * 60)
    target2 = load_or_generate_target(TAR_WAYPOINTS)
    print(f"State[0]: {target2.state}")

    # Verify identical
    assert np.allclose(target1.state, target2.state)
    assert len(target1.trajectory) == len(target2.trajectory)
    print("\n✅ Cache hoạt động đúng — trajectory load lại giống hệt")

    print("\n" + "=" * 60)
    print("Test 3: Force regen")
    print("=" * 60)
    target3 = load_or_generate_target(TAR_WAYPOINTS, force_regen=True)
    print(f"State[0]: {target3.state}")

    print("\n" + "=" * 60)
    print("List cache:")
    list_cache()