"""Native mode selection tests with fake devices; no camera is opened."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.macos_devices import configure_macos_camera_mode


class CameraModeTests(unittest.TestCase):
    def setUp(self):
        self.rate = Mock()
        self.rate.maxFrameRate.return_value = 30.00003
        self.rate.minFrameDuration.return_value = "advertised-duration"
        self.fmt = Mock()
        self.fmt.formatDescription.return_value = {"size": SimpleNamespace(width=640, height=480),
                                                   "subtype": int.from_bytes(b"420v", "big")}
        self.fmt.videoSupportedFrameRateRanges.return_value = [self.rate]
        self.device = Mock()
        self.device.localizedName.return_value = "Innomaker"
        self.device.formats.return_value = [self.fmt]
        self.device.lockForConfiguration_.return_value = (True, None)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("AVFoundation.AVCaptureDevice", SimpleNamespace(
            devicesWithMediaType_=lambda media: [self.device])))
        self.stack.enter_context(patch("CoreMedia.CMVideoFormatDescriptionGetDimensions",
                                       side_effect=lambda desc: desc["size"]))
        self.stack.enter_context(patch("CoreMedia.CMFormatDescriptionGetMediaSubType",
                                       side_effect=lambda desc: desc["subtype"]))

    def test_uses_advertised_duration_instead_of_rounded_fps(self):
        configure_macos_camera_mode("Innomaker", 640, 480, 30)
        self.device.setActiveFormat_.assert_called_once_with(self.fmt)
        self.device.setActiveVideoMaxFrameDuration_.assert_called_once_with("advertised-duration")
        self.device.unlockForConfiguration.assert_called_once()

    def test_rejects_unsupported_resolution_without_changing_device(self):
        with self.assertRaises(RuntimeError):
            configure_macos_camera_mode("Innomaker", 1280, 720, 30)
        self.device.lockForConfiguration_.assert_not_called()

    def test_rejects_unsupported_rate_without_changing_device(self):
        with self.assertRaises(RuntimeError):
            configure_macos_camera_mode("Innomaker", 640, 480, 60)
        self.device.setActiveFormat_.assert_not_called()

    def test_unlocks_after_configuration_error(self):
        self.device.setActiveFormat_.side_effect = RuntimeError("device rejected format")
        with self.assertRaises(RuntimeError):
            configure_macos_camera_mode("Innomaker", 640, 480, 30)
        self.device.unlockForConfiguration.assert_called_once()

    def test_explicit_mode_requires_complete_config(self):
        with self.assertRaises(ValueError):
            OpenCVCameraConfig(index_or_path=0, macos_explicit_mode=True)


if __name__ == "__main__":
    unittest.main()
