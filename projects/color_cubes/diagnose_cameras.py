#!/usr/bin/env python3
"""Camera-only diagnostic. Never opens arm ports or sends motor commands."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import cv2
import draccus
import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.opencv.macos_devices import list_macos_cameras
from lerobot.utils.robot_utils import precise_sleep

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["wrist", "both"], default="both")
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--native-mode", action="store_true", help="Explicitly select the AVFoundation mode after OpenCV opens")
    parser.add_argument("--wrist-width", type=int)
    parser.add_argument("--wrist-height", type=int)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("Duration must be positive")
    if (args.wrist_width is None) != (args.wrist_height is None):
        parser.error("Specify both wrist width and height")
    if args.wrist_width is not None and min(args.wrist_width, args.wrist_height) <= 0:
        parser.error("Image size must be positive")
    configs = json.loads((PROJECT / "config.json").read_text())["robot"]["cameras"]
    configs = {k: v for k, v in configs.items() if args.only == "both" or k == "wrist"}
    if args.wrist_width is not None:
        configs["wrist"].update(width=args.wrist_width, height=args.wrist_height)
    out = PROJECT / "camera_check" / f"diagnostic-{args.only}-{'native' if args.native_mode else 'default'}-{time.time_ns()}"
    out.mkdir(parents=True)
    report = {"mode": args.only, "native_mode": args.native_mode,
              "duration_requested_s": args.seconds, "arms_connected": False,
              "inventory_before": list_macos_cameras(), "config": configs, "cameras": {}}
    waits = {name: [] for name in configs}
    try:
        with ExitStack() as stack:
            cameras = {}
            for name, cfg in configs.items():
                fields = dict(cfg)
                fields.pop("type")
                cam = OpenCVCamera(draccus.decode(OpenCVCameraConfig, fields))
                stack.callback(lambda c=cam: c.disconnect() if c.is_connected else None)
                cam.connect(warmup=not args.native_mode)
                if args.native_mode:
                    import AVFoundation as av
                    import CoreMedia as cm
                    devices = [d for d in av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeVideo)
                               if str(d.localizedName()) == cfg["macos_device_name"]]
                    if len(devices) != 1:
                        raise RuntimeError("Named native device missing or ambiguous")
                    device = devices[0]
                    def describe(fmt):
                        desc = fmt.formatDescription()
                        size = cm.CMVideoFormatDescriptionGetDimensions(desc)
                        return [size.width, size.height, int(cm.CMFormatDescriptionGetMediaSubType(desc)).to_bytes(4,"big").decode()]
                    before_format = describe(device.activeFormat())
                    matches = [f for f in device.formats() if describe(f) == [cfg["width"],cfg["height"],"420v"]]
                    if not matches:
                        raise RuntimeError("Requested native 420v mode unavailable")
                    fmt = matches[0]
                    rates = [r for r in fmt.videoSupportedFrameRateRanges() if abs(r.maxFrameRate()-cfg["fps"]) < 0.01]
                    if not rates:
                        raise RuntimeError("Requested native frame rate unavailable")
                    ok, error = device.lockForConfiguration_(None)
                    if not ok:
                        raise RuntimeError(str(error))
                    try:
                        device.setActiveFormat_(fmt)
                        duration = rates[0].minFrameDuration()
                        device.setActiveVideoMinFrameDuration_(duration)
                        device.setActiveVideoMaxFrameDuration_(duration)
                    finally:
                        device.unlockForConfiguration()
                    print(name, "native format", before_format, "->", describe(device.activeFormat()), flush=True)
                    warmup_deadline = time.perf_counter() + 8
                    got_frame = False
                    while time.perf_counter() < warmup_deadline:
                        try:
                            cam.async_read(timeout_ms=1000)
                            got_frame = True
                        except TimeoutError:
                            pass
                    if not got_frame:
                        raise TimeoutError("Native format did not produce a frame")
                cameras[name] = cam
                report["cameras"][name] = {"frames": 0, "timeouts": 0, "errors": []}
            start = time.perf_counter()
            last_status = start
            while time.perf_counter() - start < args.seconds:
                tick = time.perf_counter()
                for name, cam in cameras.items():
                    status = report["cameras"][name]
                    before = time.perf_counter()
                    try:
                        frame = cam.async_read(timeout_ms=500)
                        waits[name].append((time.perf_counter() - before) * 1000)
                        status["frames"] += 1
                        if status["frames"] == 1:
                            cv2.imwrite(str(out / f"{name}-first.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        status["shape"] = list(frame.shape)
                    except Exception as exc:
                        status["timeouts"] += isinstance(exc, TimeoutError)
                        status["errors"].append({"at_s": time.perf_counter() - start, "error": str(exc)})
                        print(name, str(exc), flush=True)
                        if not cam.thread or not cam.thread.is_alive():
                            raise
                precise_sleep(max(0, 0.04 - (time.perf_counter() - tick)))
                if time.perf_counter() - last_status >= 15:
                    print({k: {m: v[m] for m in ("frames", "timeouts")} for k,v in report["cameras"].items()}, flush=True)
                    last_status = time.perf_counter()
            report["elapsed_s"] = time.perf_counter() - start
            for name,cam in cameras.items():
                if waits[name]:
                    report["cameras"][name].update(wait_p95_ms=float(np.percentile(waits[name],95)),
                                                   wait_max_ms=max(waits[name]))
                    cv2.imwrite(str(out / f"{name}-last.jpg"), cv2.cvtColor(cam.read_latest(), cv2.COLOR_RGB2BGR))
    except Exception as exc:
        report["fatal_error"] = repr(exc)
    finally:
        report["inventory_after"] = list_macos_cameras()
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("REPORT", out / "report.json", flush=True)
        print(json.dumps(report.get("cameras"), indent=2), flush=True)
        if "fatal_error" in report:
            print(report["fatal_error"], flush=True)


if __name__ == "__main__":
    main()
