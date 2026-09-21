"""Camera timeout regression tests; no physical camera is opened."""

import unittest
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


class CameraTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, read_timeout_ms=500))
        self.camera.thread = Mock()
        self.camera.thread.is_alive.return_value = True
        self.camera.new_frame_event = Mock()
        self.camera.latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        connected = patch.object(OpenCVCamera, "is_connected", new_callable=PropertyMock, return_value=True)
        connected.start()
        self.addCleanup(connected.stop)

    def test_configured_wait_allows_delayed_fresh_frame(self):
        # Emulate a frame that arrives after 300 ms: the previous default would fail.
        self.camera.new_frame_event.wait.side_effect = lambda timeout: timeout >= 0.3
        self.assertIs(self.camera.async_read(), self.camera.latest_frame)
        self.camera.new_frame_event.wait.assert_called_once_with(timeout=0.5)

    def test_explicit_timeout_overrides_configuration(self):
        self.camera.new_frame_event.wait.return_value = False
        with self.assertRaises(TimeoutError):
            self.camera.async_read(timeout_ms=0)
        self.camera.new_frame_event.wait.assert_called_once_with(timeout=0)

    def test_missing_new_frame_never_returns_stale_image(self):
        self.camera.new_frame_event.wait.return_value = False
        with self.assertRaises(TimeoutError):
            self.camera.async_read()

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            OpenCVCameraConfig(index_or_path=0, read_timeout_ms=0)

    def test_default_remains_200ms(self):
        self.assertEqual(OpenCVCameraConfig(index_or_path=0).read_timeout_ms, 200)


if __name__ == "__main__":
    unittest.main()
