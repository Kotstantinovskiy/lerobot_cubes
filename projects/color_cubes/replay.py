#!/usr/bin/env python3
"""Replay a local episode on the follower with a gradual move to its start."""

import argparse
import json
from pathlib import Path
import time

import draccus
import numpy as np
import pandas as pd

from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401

PROJECT = Path(__file__).resolve().parent


def slow_trajectory(actions, speed):
    """Keep the command rate fixed while interpolating a slower trajectory."""
    if not np.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("Speed must be greater than 0 and no greater than 1")
    positions = np.arange(int(np.ceil((len(actions) - 1) / speed)) + 1) * speed
    positions = np.minimum(positions, len(actions) - 1)
    left = positions.astype(int)
    right = np.minimum(left + 1, len(actions) - 1)
    weights = (positions - left)[:, None]
    return actions[left] * (1 - weights) + actions[right] * weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed in (0, 1]; 0.5 is half speed")
    parser.add_argument("--dry-run", action="store_true", help="Validate and describe without opening the robot")
    args = parser.parse_args()
    config = json.loads((PROJECT / "config.json").read_text())
    root = PROJECT / config["dataset_root"]
    info = json.loads((root / "meta/info.json").read_text())
    frames = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("**/*.parquet"))])
    frames = frames[frames.episode_index == args.episode].sort_values("frame_index")
    if frames.empty:
        raise ValueError("Episode not found")
    names = info["features"]["action"]["names"]
    actions = np.stack(frames.action).astype(float)
    if actions.shape != (len(frames), len(names)) or not np.isfinite(actions).all():
        raise ValueError("Invalid recorded actions")
    fps = info["fps"]
    actions = slow_trajectory(actions, args.speed)
    approach_s = 4 / args.speed
    print(f"Episode {args.episode}, speed {args.speed:g}x, {len(actions)} commands, "
          f"approximately {len(actions) / fps:.1f} seconds plus {approach_s:g} seconds approach.", flush=True)
    if args.dry_run:
        return
    fields = dict(config["robot"])
    fields.update(cameras={}, max_relative_target=10.0)
    robot = make_robot_from_config(draccus.decode(RobotConfig, fields))
    if set(names) != set(robot.action_features):
        raise ValueError("Recorded joints do not match robot")
    try:
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Calibration mismatch; refusing automatic calibration")
        if any(robot.bus.sync_read("Torque_Enable").values()):
            raise RuntimeError("Expected motors at rest with torque disabled")
        current = robot.bus.sync_read("Present_Position")
        initial = np.array([current[n.removesuffix(".pos")] for n in names])
        # Set current targets while torque is off, avoiding a jump to a stale goal on enable.
        robot.bus.sync_write("Goal_Position", current)
        robot.configure()

        def send(target):
            observation = robot.get_observation()
            errors = {n: abs(float(target[j]) - observation[n]) for j, n in enumerate(names)}
            joint = max(errors, key=errors.get)
            error = errors[joint]
            if error > 15:
                raise RuntimeError(f"{joint} is too far from next target ({error:.1f} normalized units)")
            robot.send_action(dict(zip(names, map(float, target))))

        print(f"Moving gradually to episode start ({approach_s:g} seconds).", flush=True)
        for weight in np.linspace(0, 1, int(approach_s * fps) + 1):
            tick = time.perf_counter()
            send(initial + weight * (actions[0] - initial))
            time.sleep(max(0, 1 / fps - (time.perf_counter() - tick)))
        print(f"Replaying episode {args.episode}: {len(actions)} frames at {fps} fps.", flush=True)
        start = time.perf_counter()
        for index, action in enumerate(actions):
            tick = time.perf_counter()
            send(action)
            time.sleep(max(0, 1 / fps - (time.perf_counter() - tick)))
            if (index + 1) % (10 * fps) == 0:
                print(f"Replayed {index + 1}/{len(actions)} frames.", flush=True)
        print(f"Replay completed in {time.perf_counter() - start:.2f} seconds.", flush=True)
    finally:
        if robot.bus.is_connected:
            robot.disconnect()
            print("Follower disconnected; torque disabled.", flush=True)


if __name__ == "__main__":
    main()
