"""Fixed low-level API shared by simulation and a future real Piper adapter.

Task code must only call these methods; simulator object state is inaccessible.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray  # uint8 H,W,3
    intrinsic: np.ndarray  # 3x3, pixels
    camera_to_base: np.ndarray  # 4x4, metres, OpenCV optical axes
    timestamp_s: float
    depth_m: np.ndarray | None = None  # H,W; None when no depth sensor exists


@dataclass(frozen=True)
class RobotState:
    joints_rad: np.ndarray  # J1..J6
    gripper_aperture_m: float  # continuous joint7-joint8 equivalent
    timestamp_s: float


class RobotAPI(Protocol):
    def camera_frame(self, name: str) -> CameraFrame: ...
    def robot_state(self) -> RobotState: ...
    def solve_grasp_ik(self, target_xyz_m: np.ndarray, seed_rad: np.ndarray) -> tuple[np.ndarray, float]: ...
    def move_joints(self, joints_rad: np.ndarray, aperture_m: float, duration_s: float) -> None: ...
    def close(self) -> None: ...
