"""Interface template for the other server's Piper/camera drivers.

Implement against the actual SDK and calibration. Importing this file never
connects to or commands hardware.
"""
from robot_api import CameraFrame, RobotState


class RealPiperAdapter:
    def __init__(self, piper_driver, fixed_camera, wrist_camera, calibration):
        self.piper_driver = piper_driver
        self.fixed_camera = fixed_camera
        self.wrist_camera = wrist_camera
        self.calibration = calibration

    def camera_frame(self, name: str) -> CameraFrame:
        raise NotImplementedError("Return RGB/RGBD, intrinsics, timestamp, and camera_to_base")

    def robot_state(self) -> RobotState:
        raise NotImplementedError("Read measured J1..J6 and calibrated aperture")

    def solve_grasp_ik(self, target_xyz_m, seed_rad):
        raise NotImplementedError("Use calibrated TCP and checked joint limits")

    def move_joints(self, joints_rad, aperture_m, duration_s):
        raise NotImplementedError("Bound speed, check feedback and stop conditions")

    def close(self):
        raise NotImplementedError("Release driver and camera resources")
