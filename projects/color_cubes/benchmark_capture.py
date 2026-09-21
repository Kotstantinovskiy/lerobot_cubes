#!/usr/bin/env python3
"""Measure cameras, arm reads and PNG writing without sending motor commands.

This is a capture benchmark, not a full recording or a training demonstration.
Temporary images are removed; measured timings are saved to a JSON report.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time

import draccus
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.datasets.image_writer import AsyncImageWriter
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config, so_leader  # noqa: F401
from lerobot.utils.robot_utils import precise_sleep
from record import load_config

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--read-timeout-ms", type=int, default=200)
    args = parser.parse_args()
    if min(args.fps, args.seconds, args.read_timeout_ms) <= 0:
        parser.error("All numeric arguments must be positive")
    if (args.width is None) != (args.height is None):
        parser.error("Specify both width and height or neither")
    if args.width is not None and min(args.width, args.height) <= 0:
        parser.error("Image size must be positive")
    config = load_config()
    if args.width is not None:
        for camera in config["robot"]["cameras"].values():
            camera.update(width=args.width, height=args.height)
    robot = make_robot_from_config(draccus.decode(RobotConfig, config["robot"]))
    leader = make_teleoperator_from_config(draccus.decode(TeleoperatorConfig, config["teleop"]))
    result = {"target_fps": args.fps, "camera_configs": config["robot"]["cameras"],
              "display_data": False, "motor_commands_sent": False,
              "read_timeout_ms": args.read_timeout_ms}
    with ExitStack() as stack:
        for bus in (robot.bus, leader.bus):
            bus.connect()
            stack.callback(bus.disconnect, disable_torque=False)
        for camera in robot.cameras.values():
            stack.callback(lambda c=camera: c.disconnect() if c.is_connected else None)
            camera.connect()
        temp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="cube-capture-")))
        for name in robot.cameras:
            (temp / name).mkdir()
        writer = AsyncImageWriter(num_processes=0, num_threads=8)
        stack.callback(writer.stop)
        samples = []
        stages = {name: [] for name in ["follower_read", "leader_read", *robot.cameras, "enqueue"]}
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds:
            tick = time.perf_counter()
            previous = tick
            for name, bus in (("follower_read", robot.bus), ("leader_read", leader.bus)):
                bus.sync_read("Present_Position")
                now = time.perf_counter()
                stages[name].append((now - previous) * 1000)
                previous = now
            frames = {}
            for name, camera in robot.cameras.items():
                frames[name] = camera.async_read(timeout_ms=args.read_timeout_ms)
                now = time.perf_counter()
                stages[name].append((now - previous) * 1000)
                if now - previous > 0.2:
                    print(f"Delayed fresh frame: {name}, {(now-previous)*1000:.0f} ms", flush=True)
                previous = now
            for name, frame in frames.items():
                writer.save_image(frame, temp / name / f"{count:06d}.png")
            stages["enqueue"].append((time.perf_counter() - previous) * 1000)
            # Reserve 3 ms for motor command/record metadata overhead, without moving the arms.
            precise_sleep(0.003)
            samples.append(time.perf_counter() - tick)
            precise_sleep(max(0, 1 / args.fps - (time.perf_counter() - tick)))
            count += 1
        elapsed = time.perf_counter() - start
        pending = writer.queue.qsize()
        drain = time.perf_counter()
        writer.wait_until_done()
        result.update(frames=count, elapsed_s=elapsed, actual_fps=count / elapsed,
                      work_p50_ms=float(np.percentile(samples, 50) * 1000),
                      work_p95_ms=float(np.percentile(samples, 95) * 1000),
                      over_budget_fraction=float(np.mean(np.array(samples) > 1 / args.fps)),
                      pending_images=pending, drain_s=time.perf_counter() - drain,
                      images_written=len(list(temp.glob("*/*.png"))),
                      stages_mean_ms={key: float(np.mean(value)) for key, value in stages.items()},
                      stages_max_ms={key: float(max(value)) for key, value in stages.items()},
                      camera_reads_over_200ms={key: int(np.sum(np.array(stages[key]) > 200))
                                               for key in robot.cameras})
    size = f"{args.width}x{args.height}" if args.width else "configured-named"
    report = PROJECT / "camera_check" / f"benchmark-{size}-{args.fps}fps-{args.seconds}s-{args.read_timeout_ms}ms.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
