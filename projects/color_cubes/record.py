#!/usr/bin/env python3
"""Record one color per session using the installed LeRobot recorder."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Значение должно быть больше нуля")
    return number


def build_command(config, color, episodes, resume):
    task = config["tasks"][color]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Для цвета нужна непустая текстовая команда")
    root = Path(config["dataset_root"]).expanduser()
    if not root.is_absolute():
        root = PROJECT / root
    options = {}
    for section in ("robot", "teleop"):
        for key, value in config[section].items():
            options[f"{section}.{key}"] = value
    options.update({
        "dataset.repo_id": config["repo_id"],
        "dataset.root": str(root.resolve()),
        "dataset.single_task": task,
        "dataset.num_episodes": episodes,
        "dataset.fps": config["fps"],
        "dataset.episode_time_s": config["episode_time_s"],
        "dataset.reset_time_s": config["reset_time_s"],
        "dataset.video": True,
        "dataset.vcodec": "h264",
        "dataset.push_to_hub": False,
        "display_data": config.get("display_data", True),
        "play_sounds": False,
        "resume": resume,
    })
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_record"]
    for key, value in options.items():
        encoded = json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)
        command.append(f"--{key}={encoded}")
    return command, root.resolve()


def validate_hardware(config, require_cameras=True):
    for section in ("robot", "teleop"):
        for key in ("type", "port", "id"):
            value = config[section].get(key)
            if not isinstance(value, str) or not value or "SET_" in value:
                raise ValueError(f"Заполни {section}.{key} в config.json")
    if config["robot"]["port"] == config["teleop"]["port"]:
        raise ValueError("У ведущей и ведомой руки должны быть разные порты")
    if not require_cameras:
        return
    cameras = config["robot"].get("cameras", {})
    if not cameras:
        raise ValueError("Для VLA нужна хотя бы одна камера")
    for name, camera in cameras.items():
        if camera.get("index_or_path") is None:
            raise ValueError(f"Укажи индекс камеры {name} или удали её из настроек")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--color", required=True, help="Ключ цвета из tasks в config.json")
    parser.add_argument("--episodes", type=positive_int, default=5)
    parser.add_argument("--resume", action="store_true", help="Добавить эпизоды в существующий датасет")
    parser.add_argument("--dry-run", action="store_true", help="Показать команду без подключения устройств")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        if args.color not in config["tasks"]:
            raise ValueError(f"Доступные цвета: {', '.join(config['tasks'])}")
        command, root = build_command(config, args.color, args.episodes, args.resume)
        print(f"Задание: {config['tasks'][args.color]}", flush=True)
        print(f"Датасет: {root}", flush=True)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        validate_hardware(config)
        if importlib.util.find_spec("lerobot") is None:
            raise ValueError("LeRobot не установлен в этом Python. См. раздел установки в README.md")
        if args.resume and not (root / "meta" / "info.json").is_file():
            raise ValueError("Нет существующего датасета: первый запуск сделай без --resume")
        if not args.resume and root.exists():
            raise ValueError("Папка датасета уже существует: используй --resume или другой dataset_root")
        root.parent.mkdir(parents=True, exist_ok=True)
        # Use the same interpreter/environment; never execute configuration as shell code.
        env = os.environ.copy()
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        return subprocess.call(command, env=env)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
