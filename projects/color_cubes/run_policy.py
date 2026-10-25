#!/usr/bin/env python3
"""Let a SmolVLA policy drive the follower inside a safety envelope; nothing is added to the dataset.

Observations are normalized with our dataset statistics. Every command is clipped to the joint range
seen in our demonstrations and to `--max-step` per frame. Torque is disabled on exit or Ctrl+C.
A side-by-side top/wrist video of the run is saved under camera_check/.
"""

import argparse
import functools
from pathlib import Path
import time

import av
import cv2
import draccus
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from record import PROJECT, load_config, positive_int

# Our camera keys -> the keys smolvla_base was pretrained with (camera1 = top, camera2 = wrist).
RENAME = {"observation.images.top": "observation.images.camera1",
          "observation.images.wrist": "observation.images.camera2"}


def load_policy(path, device, n_action_steps, ds_meta):
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    cfg.device = device
    cfg.n_action_steps = n_action_steps
    if hasattr(cfg, "load_vlm_weights"):
        cfg.load_vlm_weights = False  # the policy checkpoint already holds every VLM weight
    rename = RENAME if "observation.images.camera1" in cfg.input_features else {}
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map=rename)
    policy.eval()
    features = {**policy.config.input_features, **policy.config.output_features}
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=path, dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename},
            "normalizer_processor": {"stats": ds_meta.stats, "features": features,
                                     "norm_map": policy.config.normalization_mapping},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {"stats": ds_meta.stats, "features": policy.config.output_features,
                                       "norm_map": policy.config.normalization_mapping},
        },
    )
    return policy, pre, post


def dataset_observation(ds_meta, root, episode, offset_s):
    """A recorded frame in the same format as robot.get_observation(), for --dry-run."""
    dataset = LeRobotDataset(ds_meta.repo_id, root=root, video_backend="pyav")
    item = dataset[int(ds_meta.episodes[episode]["dataset_from_index"]) + int(offset_s * ds_meta.fps)]
    obs = dict(zip(ds_meta.features["observation.state"]["names"], item["observation.state"].tolist()))
    for key in ("top", "wrist"):
        image = item[f"observation.images.{key}"].permute(1, 2, 0).numpy()
        obs[key] = (image * 255).round().astype(np.uint8)
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--policy", default="lerobot/smolvla_base", help="Hub id or local pretrained_model folder")
    parser.add_argument("--color", required=True, help="Key from tasks in config.json")
    parser.add_argument("--seconds", type=positive_int, default=20)
    # 2.0 stalled the elbow against gravity (the per-frame position error was too small to move it).
    parser.add_argument("--max-step", type=float, default=5.0, help="Max change per frame per joint (normalized units)")
    parser.add_argument("--n-action-steps", type=positive_int, default=50)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dry-run", action="store_true", help="Run one inference on a recorded frame; no robot")
    args = parser.parse_args()

    config = load_config(args.config)
    task = config["tasks"][args.color]
    ds_meta = LeRobotDatasetMetadata(config["repo_id"], root=config["dataset_root"])
    names = ds_meta.features["action"]["names"]
    low, high = (np.asarray(ds_meta.stats["action"][k], dtype=float) for k in ("min", "max"))
    print(f"Task: {task}\nPolicy: {args.policy} on {args.device}\nJoint envelope from {ds_meta.total_episodes} "
          f"episodes: " + ", ".join(f"{n} [{lo:.0f}, {hi:.0f}]" for n, lo, hi in zip(names, low, high)), flush=True)
    robot = None
    if not args.dry_run:  # build the robot first so configuration errors surface before the slow model load
        fields = dict(config["robot"], max_relative_target=args.max_step)
        robot = make_robot_from_config(draccus.decode(RobotConfig, fields))
        if set(robot.calibration) != set(robot.bus.motors):
            raise RuntimeError(f"Missing or incomplete follower calibration: {robot.calibration_fpath}")
        # Torque_Enable occasionally gets no status packet right after the cameras start (seen on ids 4 and 6);
        # rewriting it is idempotent, so allow a few retries instead of aborting the connection.
        robot.bus.enable_torque = functools.partial(robot.bus.enable_torque, num_retry=3)
    policy, pre, post = load_policy(args.policy, args.device, args.n_action_steps, ds_meta)
    device = get_safe_torch_device(args.device)

    def act(obs):
        frame = build_dataset_frame(ds_meta.features, obs, prefix=OBS_STR)
        raw = make_robot_action(predict_action(frame, policy, device, pre, post, use_amp=False, task=task,
                                               robot_type=ds_meta.robot_type), ds_meta.features)
        values = np.array([raw[n] for n in names], dtype=float)
        clipped = np.clip(values, low, high)
        return dict(zip(names, clipped.tolist())), values, int((clipped != values).sum())

    if args.dry_run:
        obs = dataset_observation(ds_meta, config["dataset_root"], episode=1, offset_s=3)
        start = time.perf_counter()
        action, values, clipped = act(obs)
        print(f"Dry run: one inference {time.perf_counter() - start:.2f}s, {clipped} joints clipped.\n"
              f"  state  {np.round([obs[n] for n in names], 1).tolist()}\n  action {np.round(values, 1).tolist()}",
              flush=True)
        return

    video_path = PROJECT / "camera_check" / f"policy-run-{time.strftime('%Y%m%d-%H%M%S')}-{args.color}.mp4"
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=ds_meta.fps, options={"crf": "28"})
    stream.width, stream.height, stream.pix_fmt = 1280, 360, "yuv420p"
    stats = {"steps": 0, "chunks": 0, "max_infer_s": 0.0, "clipped_steps": 0, "late_steps": 0}
    try:
        # With torque still off: check calibration and replace any stale goal, so enabling torque cannot jump.
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Calibration mismatch; refusing automatic calibration")
        if any(robot.bus.sync_read("Torque_Enable").values()):
            raise RuntimeError("Expected motors at rest with torque disabled")
        robot.bus.sync_write("Goal_Position", robot.bus.sync_read("Present_Position"))
        robot.bus.disconnect(disable_torque=False)
        robot.connect(calibrate=False)
        print(f"Policy controls the arm for {args.seconds} s (max {args.max_step} units/frame). Ctrl+C stops.",
              flush=True)
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds:
            tick = time.perf_counter()
            obs = robot.get_observation()
            infer_start = time.perf_counter()
            action, _, clipped = act(obs)
            infer = time.perf_counter() - infer_start
            robot.send_action(action)
            stats["steps"] += 1
            stats["clipped_steps"] += clipped > 0
            stats["max_infer_s"] = max(stats["max_infer_s"], infer)
            stats["chunks"] += infer > 0.1
            side = np.hstack([cv2.resize(obs[k], (640, 360), interpolation=cv2.INTER_AREA) for k in ("top", "wrist")])
            for packet in stream.encode(av.VideoFrame.from_ndarray(side, format="rgb24")):
                container.mux(packet)
            if stats["steps"] % ds_meta.fps == 0:
                print(f"{time.perf_counter() - start:5.1f}s step {stats['steps']}, new chunks {stats['chunks']}, "
                      f"clipped steps {stats['clipped_steps']}", flush=True)
            elapsed = time.perf_counter() - tick
            stats["late_steps"] += elapsed > 1 / ds_meta.fps
            precise_sleep(max(0.0, 1 / ds_meta.fps - elapsed))
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.", flush=True)
    finally:
        if robot.bus.is_connected:  # robot.is_connected is also False when only a camera has failed
            try:
                robot.disconnect()
            except Exception:
                robot.bus.disconnect(disable_torque=True)
            print("Follower disconnected; torque disabled.", flush=True)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        print(f"Summary: {stats}\nVideo: {video_path}", flush=True)


if __name__ == "__main__":
    main()
