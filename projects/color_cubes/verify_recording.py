#!/usr/bin/env python3
"""Exercise the real recorder with arm reads only. Output is diagnostic, not training data."""

from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

from record import PROJECT, build_command, load_config, positive_int
from lerobot.scripts import lerobot_record as recorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=positive_int, default=60)
    args = parser.parse_args()
    config = load_config()
    root = PROJECT / "camera_check" / f"recording-check-{time.time_ns()}"
    config.update(dataset_root=str(root), repo_id="local/diagnostic_camera_check", episode_time_s=args.seconds,
                  display_data=False)
    config["tasks"] = {"diagnostic": "Diagnostic stationary capture. No motor commands. Not training data."}
    command, _ = build_command(config, "diagnostic", 1, False)
    make_robot = recorder.make_robot_from_config
    make_leader = recorder.make_teleoperator_from_config
    record_loop = recorder.record_loop
    report = {"motor_commands_sent": False, "target_fps": config["fps"], "dataset": str(root)}

    with ExitStack() as stack:
        def connect_bus(bus):
            bus.connect()
            stack.callback(bus.disconnect, disable_torque=False)

        def read_only_robot(cfg):
            robot = make_robot(cfg)

            def connect():
                connect_bus(robot.bus)
                for camera in robot.cameras.values():
                    stack.callback(lambda c=camera: c.disconnect() if c.is_connected else None)
                    camera.connect()

            robot.connect = connect
            robot.disconnect = lambda: None  # ExitStack closes camera streams and serial ports.
            robot.send_action = lambda action: action  # Never writes a goal position or enables torque.
            return robot

        def read_only_leader(cfg):
            leader = make_leader(cfg)
            leader.connect = lambda: connect_bus(leader.bus)
            leader.disconnect = lambda: None
            return leader

        def measured_loop(*args, **kwargs):
            start = time.perf_counter()
            result = record_loop(*args, **kwargs)
            report["wall_time_s"] = time.perf_counter() - start
            report["frames"] = kwargs["dataset"].episode_buffer["size"]
            report["actual_fps"] = report["frames"] / report["wall_time_s"]
            return result

        stack.enter_context(patch.object(recorder, "make_robot_from_config", read_only_robot))
        stack.enter_context(patch.object(recorder, "make_teleoperator_from_config", read_only_leader))
        stack.enter_context(patch.object(recorder, "record_loop", measured_loop))
        # Keyboard input is deliberately disabled for this timed, stationary diagnostic.
        stack.enter_context(patch.object(recorder, "init_keyboard_listener", return_value=(None, {
            "stop_recording": False, "exit_early": False, "rerecord_episode": False})))
        stack.enter_context(patch.object(sys, "argv", ["verify_recording", *command[3:]]))
        recorder.main()
    (root / "diagnostic-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
