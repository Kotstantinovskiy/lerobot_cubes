"""Resolve camera names in the same order as OpenCV 4.12's AVFoundation backend.

OpenCV concatenates video and muxed devices, then sorts by uniqueID:
https://github.com/opencv/opencv/blob/4.12.0/modules/videoio/src/cap_avfoundation_mac.mm
FFmpeg's displayed indices must not be used as OpenCV indices.
"""

import platform


def list_macos_cameras() -> list[dict]:
    if platform.system() != "Darwin":
        raise RuntimeError("Named AVFoundation cameras require macOS")
    try:
        import AVFoundation as av
        from Foundation import NSSortDescriptor, NSArray
    except ImportError as exc:
        raise RuntimeError("Install pyobjc-framework-AVFoundation to select cameras by name") from exc
    devices = list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeVideo))
    devices += list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeMuxed))
    # Native NSString comparison, matching the OpenCV backend's sort order.
    ordered = NSArray.arrayWithArray_(devices).sortedArrayUsingDescriptors_(
        [NSSortDescriptor.sortDescriptorWithKey_ascending_("uniqueID", True)]
    )
    return [{"index": i, "name": str(d.localizedName()), "uid": str(d.uniqueID())}
            for i, d in enumerate(ordered)]


def resolve_macos_camera(name: str) -> dict:
    devices = list_macos_cameras()
    matches = [device for device in devices if device["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one camera named {name!r}, found {len(matches)}. "
                           f"Available: {[d['name'] for d in devices]}")
    return matches[0]


def configure_macos_camera_mode(name: str, width: int, height: int, fps: float) -> dict:
    """Explicitly select a supported native NV12 format and its advertised frame duration.

    Called after OpenCV opens the session, before its read thread starts. Do not
    assume that the output pixel format alone selects the camera's input mode.
    """
    import AVFoundation as av
    import CoreMedia as cm

    devices = [d for d in av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeVideo)
               if str(d.localizedName()) == name]
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one camera named {name!r}")
    device = devices[0]
    selected = None
    for fmt in device.formats():
        desc = fmt.formatDescription()
        size = cm.CMVideoFormatDescriptionGetDimensions(desc)
        subtype = int(cm.CMFormatDescriptionGetMediaSubType(desc))
        if (size.width, size.height, subtype) != (width, height, int.from_bytes(b"420v", "big")):
            continue
        for rate in fmt.videoSupportedFrameRateRanges():
            if abs(rate.maxFrameRate() - fps) < 0.01:
                selected = (fmt, rate)
                break
        if selected:
            break
    if selected is None:
        raise RuntimeError(f"No native 420v format for {name!r}: {width}x{height} at {fps} fps")
    ok, error = device.lockForConfiguration_(None)
    if not ok:
        raise RuntimeError(f"Cannot configure camera {name!r}: {error}")
    try:
        fmt, rate = selected
        device.setActiveFormat_(fmt)
        duration = rate.minFrameDuration()
        device.setActiveVideoMinFrameDuration_(duration)
        device.setActiveVideoMaxFrameDuration_(duration)
    finally:
        device.unlockForConfiguration()
    return {"name": name, "width": width, "height": height, "pixel_format": "420v",
            "advertised_fps": float(rate.maxFrameRate())}
