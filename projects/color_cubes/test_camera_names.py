import unittest
from unittest.mock import patch

from lerobot.cameras.opencv.macos_devices import resolve_macos_camera


class CameraNamesTests(unittest.TestCase):
    def test_follows_name_when_index_changes(self):
        devices = [{"index": 0, "name": "FaceTime HD Camera", "uid": "a"},
                   {"index": 2, "name": "Innomaker-U20CAM-1080p-S1", "uid": "b"}]
        with patch("lerobot.cameras.opencv.macos_devices.list_macos_cameras", return_value=devices):
            self.assertEqual(resolve_macos_camera("Innomaker-U20CAM-1080p-S1"), devices[1])

    def test_missing_camera_does_not_fall_back_to_webcam(self):
        with patch("lerobot.cameras.opencv.macos_devices.list_macos_cameras", return_value=[
            {"index": 1, "name": "FaceTime HD Camera", "uid": "a"}
        ]):
            with self.assertRaises(RuntimeError):
                resolve_macos_camera("Innomaker-U20CAM-1080p-S1")

    def test_ambiguous_name_rejected(self):
        with patch("lerobot.cameras.opencv.macos_devices.list_macos_cameras", return_value=[
            {"index": 0, "name": "Sport Cam", "uid": "a"},
            {"index": 1, "name": "Sport Cam", "uid": "b"}
        ]):
            with self.assertRaises(RuntimeError):
                resolve_macos_camera("Sport Cam")


if __name__ == "__main__":
    unittest.main()
