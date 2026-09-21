#!/usr/bin/env python3
"""Sequential wrist-camera A/B capture test; never connects to robot motors."""

import argparse
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parent


def worker(backend, out, seconds, with_top=False):
    import cv2
    import numpy as np
    from lerobot.cameras.opencv.macos_devices import resolve_macos_camera

    cfg = json.loads((PROJECT / "config.json").read_text())["robot"]["cameras"]["wrist"]
    identity = resolve_macos_camera(cfg["macos_device_name"])
    result = {"backend": backend, "identity_before": identity, "config": cfg,
              "seconds_requested": seconds, "frames": 0, "timeouts": 0}
    result["with_top"] = with_top
    cam = proc = top = top_thread = None
    top_stop = threading.Event()
    measuring = threading.Event()
    top_stats = {"frames": 0, "timeouts": 0, "max_gap_ms": 0, "gaps_over_500ms": 0}
    log = None
    gaps = []
    started = time.monotonic()
    try:
        if with_top:
            import draccus
            from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
            fields = json.loads((PROJECT / "config.json").read_text())["robot"]["cameras"]["top"]
            result["top_identity"] = resolve_macos_camera(fields["macos_device_name"])
            fields.pop("type")
            top = OpenCVCamera(draccus.decode(OpenCVCameraConfig, fields))
            top.connect()

            def monitor_top():
                previous = None
                last = None
                while not top_stop.is_set():
                    try:
                        frame = top.async_read(timeout_ms=500)
                    except TimeoutError:
                        if measuring.is_set():
                            top_stats["timeouts"] += 1
                        continue
                    except Exception as exc:
                        top_stats["error"] = repr(exc)
                        break
                    if not measuring.is_set():
                        continue
                    now = time.monotonic()
                    if previous is not None:
                        gap = (now - previous) * 1000
                        top_stats["max_gap_ms"] = max(top_stats["max_gap_ms"], gap)
                        top_stats["gaps_over_500ms"] += gap > 500
                    previous = now
                    last = frame
                    top_stats["frames"] += 1
                    if top_stats["frames"] == 1:
                        cv2.imwrite(str(out / "top-first.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if last is not None:
                    cv2.imwrite(str(out / "top-last.jpg"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))

            top_thread = threading.Thread(target=monitor_top, daemon=True)
            top_thread.start()
            started = time.monotonic()
        if backend == "lerobot":
            import draccus
            from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
            fields = dict(cfg)
            fields.pop("type")
            cam = OpenCVCamera(draccus.decode(OpenCVCameraConfig, fields))
            cam.connect()
            begin = previous = time.monotonic()
            measuring.set()
            result["startup_s"] = begin - started
            last = None
            while time.monotonic() - begin < seconds:
                try:
                    last = cam.async_read(timeout_ms=500)
                except TimeoutError:
                    result["timeouts"] += 1
                    continue
                now = time.monotonic()
                gaps.append(now - previous)
                previous = now
                result["frames"] += 1
                if result["frames"] == 1:
                    cv2.imwrite(str(out / "first.jpg"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
            result["elapsed_s"] = time.monotonic() - begin
            if last is not None:
                cv2.imwrite(str(out / "last.jpg"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
                result["shape"] = list(last.shape)
        else:
            # Match the device's advertised duration, also used by our native-mode helper.
            import AVFoundation as av
            import CoreMedia as cm
            devices = [d for d in av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeVideo)
                       if str(d.uniqueID()) == identity["uid"]]
            duration = None
            for fmt in devices[0].formats():
                desc = fmt.formatDescription()
                size = cm.CMVideoFormatDescriptionGetDimensions(desc)
                if (size.width, size.height, int(cm.CMFormatDescriptionGetMediaSubType(desc))) != (
                        cfg["width"], cfg["height"], int.from_bytes(b"420v", "big")):
                    continue
                for rate in fmt.videoSupportedFrameRateRanges():
                    if abs(rate.maxFrameRate() - cfg["fps"]) < .01:
                        duration = rate.minFrameDuration()
                        break
            if duration is None:
                raise RuntimeError("No matching native frame duration")
            cmd = ["/opt/homebrew/bin/ffmpeg", "-hide_banner", "-nostdin", "-nostats",
                   "-f", "avfoundation", "-framerate", f"{duration.timescale}/{duration.value}",
                   "-video_size", f"{cfg['width']}x{cfg['height']}", "-pixel_format", "nv12",
                   "-i", cfg["macos_device_name"] + ":none", "-an", "-fps_mode", "passthrough",
                   "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
            result["command"] = cmd
            log = (out / "ffmpeg.log").open("w")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, bufsize=0)
            os.set_blocking(proc.stdout.fileno(), False)
            buf = bytearray()
            frame_bytes = cfg["width"] * cfg["height"] * 3
            first = begin = previous = last_arrival = None
            last = None
            with selectors.DefaultSelector() as sel:
                sel.register(proc.stdout, selectors.EVENT_READ)
                while True:
                    now = time.monotonic()
                    if begin is not None and now - begin >= seconds:
                        break
                    if first is None and now - started > 15:
                        raise TimeoutError("FFmpeg produced no first frame within 15 seconds")
                    if first is not None and begin is None and now - first >= cfg["warmup_s"]:
                        begin = previous = now
                        measuring.set()
                        result["startup_s"] = now - started
                    if proc.poll() is not None:
                        raise RuntimeError(f"FFmpeg exited early: {proc.returncode}")
                    if not sel.select(.1):
                        if last_arrival is not None and now - last_arrival > 8:
                            raise TimeoutError("FFmpeg stopped delivering frames for 8 seconds")
                        continue
                    chunk = os.read(proc.stdout.fileno(), 1048576)
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    while len(buf) >= frame_bytes:
                        raw = bytes(buf[:frame_bytes])
                        del buf[:frame_bytes]
                        now = time.monotonic()
                        last_arrival = now
                        if first is None:
                            first = now
                            result["first_frame_s"] = now - started
                        if begin is None:
                            continue
                        gaps.append(now - previous)
                        previous = now
                        result["frames"] += 1
                        last = np.frombuffer(raw, dtype=np.uint8).reshape(cfg["height"], cfg["width"], 3)
                        if result["frames"] == 1:
                            cv2.imwrite(str(out / "first.jpg"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
            result["elapsed_s"] = time.monotonic() - begin
            if last is not None:
                cv2.imwrite(str(out / "last.jpg"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
                result["shape"] = list(last.shape)
        if gaps:
            result.update(fps=result["frames"] / result["elapsed_s"],
                          max_gap_ms=max(gaps) * 1000,
                          gaps_over_500ms=sum(g > .5 for g in gaps))
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        measuring.clear()
        top_stop.set()
        if top_thread is not None:
            top_thread.join(timeout=2)
            result["top"] = top_stats
        if cam is not None and cam.is_connected:
            cam.disconnect()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            proc.stdout.close()
        if top is not None and top.is_connected:
            top.disconnect()
        if log is not None:
            log.close()
            content = (out / "ffmpeg.log").read_text()
            result["ffmpeg_mode_warnings"] = [s for s in content.splitlines()
                if any(w in s.lower() for w in ("fallback", "not supported", "not support", "selected framerate", "selected video size"))]
        result["identity_after"] = resolve_macos_camera(cfg["macos_device_name"])
        (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", choices=("lerobot", "ffmpeg"))
    p.add_argument("--out", type=Path)
    p.add_argument("--seconds", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--with-top", action="store_true", help="Keep top camera streaming through our code during each wrist test")
    args = p.parse_args()
    if args.worker:
        worker(args.worker, args.out, args.seconds, args.with_top)
        return
    out = PROJECT / "camera_check" / f"backend-comparison-{time.time_ns()}"
    out.mkdir(parents=True)
    (out / "usb-before.txt").write_text(subprocess.check_output(["ioreg", "-p", "IOUSB", "-w", "0"], text=True))
    print("OUTPUT", out, flush=True)
    results = []
    for round_id in range(1, args.rounds + 1):
        for backend in ("lerobot", "ffmpeg"):
            trial = out / f"{round_id}-{backend}"
            trial.mkdir()
            print("START", trial.name, flush=True)
            with (trial / "worker.log").open("w") as log:
                try:
                    completed = subprocess.run([sys.executable, __file__, "--worker", backend,
                        "--out", str(trial), "--seconds", str(args.seconds)] +
                        (["--with-top"] if args.with_top else []), stdout=log, stderr=log,
                        timeout=args.seconds + 40)
                    report = json.loads((trial / "report.json").read_text())
                    report["exit_code"] = completed.returncode
                except Exception as exc:
                    report = {"backend": backend, "error": repr(exc)}
            report["trial"] = trial.name
            results.append(report)
            (out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({k: report[k] for k in ("trial", "frames", "fps", "max_gap_ms", "timeouts", "error", "ffmpeg_mode_warnings", "top") if k in report}), flush=True)
            time.sleep(2)
    (out / "usb-after.txt").write_text(subprocess.check_output(["ioreg", "-p", "IOUSB", "-w", "0"], text=True))
    print("DONE", out / "report.json", flush=True)


if __name__ == "__main__":
    main()
