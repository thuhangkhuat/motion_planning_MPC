"""
Run the multi-UAV MPC-CBF target-tracking simulation.

    python main.py --scenario 6
    python main.py --scenario 6 --seed 1 --max-steps 300 --quiet
    python main.py --scenario 6 --target-mode rrt --force-regen

Each run writes to runs/<name>/ :
    data.pkl      simulation data (same format as before, readable by plot_*.py)
    config.json   snapshot of parameters + scenario + seed + git commit
    log.txt       full log (also when --quiet)
By default a copy is also written to the legacy path (FILE_NAME in config) so
plot_result.py / plot_animation.py keep working; disable with --no-legacy-copy.
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", help="scenario number or name in scenarios/ "
                                      "(default: env SCENARIO or config)")
    p.add_argument("--seed", type=int, default=0, help="numpy seed (default 0)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after N steps (default: until the target reaches its goal)")
    p.add_argument("--out", default=None, help="run directory (default runs/scen<N>_..._<timestamp>)")
    p.add_argument("--tag", default="", help="suffix appended to the run directory name")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="save a checkpoint every N steps (0 = off)")
    p.add_argument("--target-mode", choices=["manual", "rrt"], default=None)
    p.add_argument("--force-regen", action="store_true", help="(rrt) regenerate the target trajectory")
    p.add_argument("--quiet", action="store_true", help="console shows only warnings + progress bar")
    p.add_argument("--ignore-config-errors", action="store_true",
                   help="run even if validate_config() reports errors")
    p.add_argument("--no-legacy-copy", action="store_true",
                   help="do not write the extra data copy to the legacy path")
    return p.parse_args()


# config reads SCENARIO / TARGET_MODE from env AT IMPORT TIME -> set them before importing
ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    if ARGS.scenario is not None:
        os.environ["SCENARIO"] = str(ARGS.scenario)
    if ARGS.target_mode is not None:
        os.environ["TARGET_MODE"] = ARGS.target_mode

import json          # noqa: E402
import logging       # noqa: E402
import pickle        # noqa: E402
import signal        # noqa: E402
import subprocess    # noqa: E402
import time          # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np   # noqa: E402

import config        # noqa: E402
from config import (NUM_ROBOT, STARTS, GOALS, SCENARIO_NAME, SCENARIO_DEF, METHOD,  # noqa: E402
                    VIEWING_RADIUS, FILE_NAME, TARGET_ARRIVAL_TOL, validate_config)

log = logging.getLogger("main")


# ============================================================
# Helpers
# ============================================================
def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def _git_commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"],
                                stderr=subprocess.DEVNULL) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return None


def config_snapshot(args, target):
    params = {k: _jsonable(v) for k, v in vars(config).items()
              if k.isupper() and k != "SCENARIO_DEF" and not callable(v)
              and not isinstance(v, type(os))}
    snap = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "scenario": SCENARIO_NAME,
        "scenario_def": _jsonable(SCENARIO_DEF),
        "target_mode": os.environ.get("TARGET_MODE", "manual"),
        "target_waypoints": _jsonable(getattr(target, "waypoints_2d", None)),
        "target_speeds": _jsonable(getattr(target, "speeds", None)),
        "params": params,
    }
    return snap


def setup_run_dir(args):
    if args.out:
        run_dir = args.out
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = f"_{args.tag}" if args.tag else ""
        run_dir = os.path.join("runs", f"{SCENARIO_NAME}_n{NUM_ROBOT}_s{args.seed}_{stamp}{tag}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logging(run_dir, quiet):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(os.path.join(run_dir, "log.txt"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    root.addHandler(ch)


def _as_array(seq):
    """Plain np.array; fall back to an object array for ragged items (numpy>=1.24 raises)."""
    try:
        return np.array(seq)
    except ValueError:
        out = np.empty(len(seq), dtype=object)
        out[:] = list(seq)
        return out


def collect_data(robots, target_traj, compute_times, n_iter, finished):
    data = {}
    for i in range(NUM_ROBOT):
        data[i] = {
            "path": np.array(robots[i].path),
            "traj_refs": _as_array(robots[i].traj_refs),
            "tar_traj": np.array(target_traj),
            "corridors": _as_array(robots[i].corridors_plot),
        }
    ct = np.array(compute_times)
    data["meta"] = {
        "compute_times": ct,
        "compute_time_mean": float(ct.mean()) if ct.size else float("nan"),
        "compute_time_max": float(ct.max()) if ct.size else float("nan"),
        "compute_time_min": float(ct.min()) if ct.size else float("nan"),
        "num_robot": NUM_ROBOT,
        "scenario": SCENARIO_NAME,
        "method": METHOD,
        "iterations": n_iter,
        "finished": finished,
        "fov_side": 2 * VIEWING_RADIUS,
    }
    return data


def atomic_pickle(obj, path):
    """Write to a temp file then rename, so a kill mid-write never leaves a corrupt file."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# ============================================================
# Main
# ============================================================
def main(args):
    np.random.seed(args.seed)

    run_dir = setup_run_dir(args)
    setup_logging(run_dir, args.quiet)

    # Parameter consistency check (see config.validate_config)
    issues = validate_config()
    for level, msg in issues:
        log.log(logging.INFO if level == "INFO" else logging.WARNING, "[config] %s", msg)
    errors = [m for lvl, m in issues if lvl == "ERROR"]
    if errors and not args.ignore_config_errors:
        log.error("Config has %d error(s); fix them or pass --ignore-config-errors.", len(errors))
        for m in errors:
            log.error("[config] %s", m)
        sys.exit(2)

    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    from robot_jps import Robot
    from target_manual import create_target

    target = create_target(force_regen=args.force_regen, verbose=not args.quiet)
    import config as _cfg
    if _cfg.PLANNER == "local":
        from planner_grid import warmup
        warmup()          # compile the numba kernel before timing starts
    robots = [Robot(i, np.concatenate([STARTS[i, :], [0, 0, 0]]), GOALS[i, :])
              for i in range(NUM_ROBOT)]

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_snapshot(args, target), f, indent=2, ensure_ascii=False)
    log.info("Run dir: %s | scenario %s | %d UAV | seed %d",
             run_dir, SCENARIO_NAME, NUM_ROBOT, args.seed)

    data_path = os.path.join(run_dir, "data.pkl")
    target_traj, compute_times = [], []
    n_iter, finished = 0, False
    total = len(getattr(target, "trajectory", [])) or None
    if args.max_steps:
        total = min(total, args.max_steps) if total else args.max_steps

    # Ctrl+C: IPOPT swallows SIGINT during a solve (it shows up as "MPC solve failed"
    # and the loop keeps going) -> catch it ourselves, set a flag, exit cleanly at the
    # end of the step. A second Ctrl+C stops immediately.
    stop = {"flag": False}

    def _on_sigint(signum, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        log.warning("Ctrl+C received — stopping and saving after the current step (press again to stop now).")

    old_handler = signal.signal(signal.SIGINT, _on_sigint)

    t_start = time.time()
    try:
        with logging_redirect_tqdm(), tqdm(total=total, unit="step", dynamic_ncols=True) as bar:
            while True:
                target.update()
                target_traj.append(target.state.copy())

                for r in robots:
                    t0 = time.perf_counter()
                    r.goal = target.state.copy()
                    r.computeControlSignal(robots)
                    compute_times.append(time.perf_counter() - t0)

                n_iter += 1
                bar.update(1)
                if n_iter % 10 == 0:
                    modes = "".join("L" if r.is_leader_role else r.mode[0] for r in robots)
                    bar.set_postfix(ms=f"{1000 * np.mean(compute_times[-10 * NUM_ROBOT:]):.0f}",
                                    mode=modes)

                if args.checkpoint_every and n_iter % args.checkpoint_every == 0:
                    atomic_pickle(collect_data(robots, target_traj, compute_times,
                                               n_iter, False), data_path)
                    log.info("Checkpoint at step %d", n_iter)

                if np.linalg.norm(target.state - target.final_destination) < TARGET_ARRIVAL_TOL:
                    finished = True
                    log.info("Target reached its final destination after %d steps.", n_iter)
                    break
                if args.max_steps and n_iter >= args.max_steps:
                    log.info("Reached --max-steps=%d.", args.max_steps)
                    break
                if stop["flag"]:
                    log.warning("Stopped by Ctrl+C at step %d.", n_iter)
                    break
    except KeyboardInterrupt:
        log.warning("Interrupted at step %d — saving data anyway.", n_iter)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        data = collect_data(robots, target_traj, compute_times, n_iter, finished)
        atomic_pickle(data, data_path)
        if not args.no_legacy_copy:
            atomic_pickle(data, FILE_NAME)

        ct = np.array(compute_times)
        wall = time.time() - t_start
        summary = (f"{n_iter} steps in {wall:.1f} s | compute per UAV per step: "
                   + (f"mean {1000 * ct.mean():.1f} ms, max {1000 * ct.max():.1f} ms"
                      if ct.size else "n/a"))
        infeas = {r.index: len(getattr(r, "infeasible_log", [])) for r in robots}
        log.info(summary)
        log.info("MPC infeasible count per UAV: %s", infeas)
        print(f"\n[DONE] {summary}\n       Data: {data_path}"
              + ("" if args.no_legacy_copy else f"  (+ legacy copy: {FILE_NAME})"))
    return data


if __name__ == "__main__":
    main(ARGS)
