#!/usr/bin/env python3
"""Управление ведомой рукой через ведущую, без записи датасета."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys

from record import PROJECT, positive_int, validate_hardware


def build_command(config, cameras=False, seconds=None):
    options = {}
    for section in ("robot", "teleop"):
        for key, value in config[section].items():
            if section == "robot" and key == "cameras" and not cameras:
                continue
            options[f"{section}.{key}"] = value
    if not cameras:
        options["robot.cameras"] = {}
    options["fps"] = config.get("fps", 30)
    options["display_data"] = cameras
    if seconds is not None:
        options["teleop_time_s"] = seconds
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_teleoperate"]
    for key, value in options.items():
        encoded = json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)
        command.append(f"--{key}={encoded}")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--cameras", action="store_true", help="Подключить камеры и показать изображение")
    parser.add_argument("--seconds", type=positive_int, help="Завершить управление через указанное число секунд")
    parser.add_argument("--dry-run", action="store_true", help="Показать команду без подключения оборудования")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        command = build_command(config, args.cameras, args.seconds)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        validate_hardware(config, require_cameras=args.cameras)
        if importlib.util.find_spec("lerobot") is None:
            raise ValueError("LeRobot не установлен в этом Python. См. раздел установки в README.md")
        print("Телеуправление без записи. Остановка: Ctrl+C.", flush=True)
        # Replace the launcher so Ctrl+C reaches LeRobot directly.
        os.execv(sys.executable, command)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
