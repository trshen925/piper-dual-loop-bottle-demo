"""MuJoCo implementation of the fixed RobotAPI.

Task code receives CameraFrame/RobotState/IK/motion only. Ground-truth object
pose is confined to reset and evaluation methods of this adapter.
"""
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from robot_api import CameraFrame, RobotState

ROOT = Path(__file__).resolve().parent


class MuJoCoRobot:
    def __init__(self, scene=ROOT / "scene/scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.qa = np.array([self.model.jnt_qposadr[self.model.joint(f"joint{i}").id] for i in range(1, 9)])
        self.va = np.array([self.model.jnt_dofadr[self.model.joint(f"joint{i}").id] for i in range(1, 9)])
        self.bq = int(self.model.jnt_qposadr[self.model.joint("bottle_free").id])
        self.bv = int(self.model.jnt_dofadr[self.model.joint("bottle_free").id])
        self.pads = [self.model.geom("pad7").id, self.model.geom("pad8").id]
        self.fingers = [self.model.geom("finger_collision7").id,
                        self.model.geom("finger_collision8").id]
        self.bottle_geom = self.model.geom("bottle_proxy").id
        self.coaster_geom = self.model.geom("white_pad").id
        self.renderer = mujoco.Renderer(self.model, 480, 640)
        self.metrics = []
        self.reset((.2882762818, .1067295579))

    def reset(self, bottle_xy_m):
        mujoco.mj_resetData(self.model, self.data)
        # Match the validated program baseline's reachable pre-grasp pose.
        self.data.qpos[self.qa] = [.354588181, 1.64865496, -1.22780676,
                                   .000007856, 1.22, 1.59976698, .035, -.035]
        self.data.qpos[self.bq:self.bq + 2] = bottle_xy_m
        mujoco.mj_forward(self.model, self.data)
        self.metrics.clear()
        self.previous_command = self.data.qpos[self.qa].copy()

    def camera_frame(self, name):
        camera = {"fixed": "camera2_fixed", "wrist": "camera1_wrist"}[name]
        cid = self.model.camera(camera).id
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        transform = np.eye(4)
        transform[:3, :3] = self.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
        transform[:3, 3] = self.data.cam_xpos[cid]
        # Explicit OpenCV intrinsics from the paired camera calibration.
        if camera == "camera2_fixed":
            fx, fy, cx, cy = 486.93767377, 483.11118452, 319.2534461, 215.27324733
        else:
            fx, fy, cx, cy = 502.20649532, 501.48448009, 325.02677741, 201.84400623
        intrinsic = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        return CameraFrame(rgb=rgb, depth_m=depth, intrinsic=intrinsic,
                           camera_to_base=transform, timestamp_s=float(self.data.time))

    def robot_state(self):
        return RobotState(joints_rad=self.data.qpos[self.qa[:6]].copy(),
                          gripper_aperture_m=float(self.data.qpos[self.qa[6]] - self.data.qpos[self.qa[7]]),
                          timestamp_s=float(self.data.time))

    def solve_grasp_ik(self, target_xyz_m, seed_rad):
        original = self.data.qpos.copy()

        def residual(q):
            self.data.qpos[self.qa] = np.r_[q, .035, -.035]
            mujoco.mj_forward(self.model, self.data)
            rotation = self.data.geom_xmat[self.pads[0]].reshape(3, 3)
            return np.r_[10 * (self.data.geom_xpos[self.pads].mean(0) - target_xyz_m),
                         rotation[:, 1] - [0, 0, -1], .0001 * (q - seed_rad)]

        result = least_squares(residual, seed_rad,
                               bounds=(self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1]),
                               max_nfev=400)
        self.data.qpos[:] = original
        mujoco.mj_forward(self.model, self.data)
        return result.x, float(np.linalg.norm(result.fun[:3]) / 10)

    def move_joints(self, joints_rad, aperture_m, duration_s):
        joints_rad = np.asarray(joints_rad, dtype=float)
        if joints_rad.shape != (6,) or not np.isfinite(joints_rad).all():
            raise ValueError("expected finite J1..J6 radians")
        if not .0 <= aperture_m <= .070:
            raise ValueError("gripper aperture outside 0..70 mm")
        if np.any(joints_rad < self.model.jnt_range[:6, 0]) or np.any(joints_rad > self.model.jnt_range[:6, 1]):
            raise ValueError("joint target outside MuJoCo limits")
        # Match the validated position-controller trajectory: each quintic
        # segment starts from the previous commanded target, not lagged qpos.
        start = self.previous_command.copy()
        goal = np.r_[joints_rad, aperture_m / 2, -aperture_m / 2]
        count = round(duration_s / self.model.opt.timestep)
        for k in range(count):
            a = (k + 1) / count
            a = a ** 3 * (10 - 15 * a + 6 * a * a)
            self.data.ctrl[:] = (1 - a) * start + a * goal
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.va[:6]] = self.data.qfrc_bias[self.va[:6]]
            mujoco.mj_step(self.model, self.data)
            forces = np.zeros(2)
            support = 0.
            penetration = 0.
            for ci in range(self.data.ncon):
                contact = self.data.contact[ci]
                penetration = max(penetration, -float(contact.dist))
                pair = {contact.geom1, contact.geom2}
                if self.bottle_geom in pair:
                    if self.coaster_geom in pair or any(f in pair for f in self.fingers):
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, ci, wrench)
                        if self.coaster_geom in pair:
                            support += wrench[0]
                        for j, finger in enumerate(self.fingers):
                            if finger in pair:
                                forces[j] += wrench[0]
            self.metrics.append([self.data.time, *self.data.qpos[self.bq:self.bq + 3],
                                 np.linalg.norm(self.data.qvel[self.bv:self.bv + 3]),
                                 *forces, support, penetration])
        self.previous_command = goal.copy()

    def evaluation(self):
        a = np.asarray(self.metrics)
        return {"final_bottle_xyz_m": self.data.qpos[self.bq:self.bq + 3].tolist(),
                "max_lift_m": float(a[:, 3].max() + .015),
                "max_penetration_m": float(a[:, 8].max()),
                "warnings": self.data.warning.number.tolist()}

    def close(self):
        self.renderer.close()
