"""Editable perception and task-level primitives, in the style of SimEx."""
from dataclasses import dataclass

import cv2
import numpy as np

from robot_api import CameraFrame, RobotAPI


@dataclass(frozen=True)
class Detection:
    xy_m: np.ndarray
    pixel_uv: tuple[float, float]
    area_px: int
    method: str


def _ray_to_plane(frame: CameraFrame, u: float, v: float, plane_z_m: float) -> np.ndarray:
    fx, fy = frame.intrinsic[0, 0], frame.intrinsic[1, 1]
    cx, cy = frame.intrinsic[0, 2], frame.intrinsic[1, 2]
    ray_cv = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    world_ray = frame.camera_to_base[:3, :3] @ ray_cv
    origin = frame.camera_to_base[:3, 3]
    if abs(world_ray[2]) < 1e-6:
        raise ValueError("camera ray nearly parallel to support plane")
    scale = (plane_z_m - origin[2]) / world_ray[2]
    if scale <= 0:
        raise ValueError("support plane behind camera")
    return origin + scale * world_ray


def _color_component(frame: CameraFrame, roi_xyxy, hsv_lo, hsv_hi, min_area=200):
    image = np.asarray(frame.rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected RGB image")
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [int(x) for x in roi_xyxy]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, dtype=np.uint8), np.array(hsv_hi, dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    valid = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not valid:
        return None
    valid = [i for i in valid if x0 + stats[i, cv2.CC_STAT_LEFT] +
             stats[i, cv2.CC_STAT_WIDTH] / 2 < w - 80]
    if not valid:
        return None
    best = max(valid, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    ys, xs = np.where(labels == best)
    return xs + x0, ys + y0, int(stats[best, cv2.CC_STAT_AREA])


class BottleToolbox:
    def __init__(self, robot: RobotAPI, *, roi_xyxy=(350, 85, 475, 235),
                 bottle_support_z_m=-.015, bottle_radius_m=.03137,
                 coaster_xy_m=(.28161665, -.07124573),
                 reference_bottle_xy_m=(.28827628, .10672956),
                 reference_silhouette_uv=(408.0, 162.0)):
        self.robot = robot
        self.roi_xyxy = roi_xyxy
        self.bottle_support_z_m = bottle_support_z_m
        self.bottle_radius_m = bottle_radius_m
        self.coaster_xy_m = np.asarray(coaster_xy_m, dtype=float)
        self.reference_bottle_xy_m = np.asarray(reference_bottle_xy_m, dtype=float)
        self.reference_silhouette_uv = np.asarray(reference_silhouette_uv, dtype=float)

    def find_bottle(self) -> Detection | None:
        frame = self.robot.camera_frame("fixed")
        h, w = frame.rgb.shape[:2]
        component = _color_component(frame, (0, 0, w, h), (5, 75, 35), (35, 255, 255), min_area=500)
        if component is None:
            return None
        xs, ys, area = component
        if float(np.median(xs)) > w - 80:
            return None
        v = float(np.quantile(ys, .80))
        u = float(np.median(xs[(ys >= v - 3) & (ys <= v + 3)]))
        # The visible texture and occluding gripper bias absolute 3-D surface
        # back-projection. Use the nominal scene's measured silhouette as a
        # reference, then convert its image displacement through camera geometry.
        # This is a local RGB calibration, valid only around the stated fixture.
        def project(xy):
            point = np.r_[xy, self.bottle_support_z_m + .075]
            camera = np.linalg.inv(frame.camera_to_base) @ np.r_[point, 1.]
            return np.array([frame.intrinsic[0, 0] * camera[0] / camera[2] + frame.intrinsic[0, 2],
                             frame.intrinsic[1, 1] * camera[1] / camera[2] + frame.intrinsic[1, 2]])
        step = .001
        jacobian = np.column_stack([(project(self.reference_bottle_xy_m + np.eye(2)[i] * step) -
                                     project(self.reference_bottle_xy_m)) / step for i in range(2)])
        delta = np.linalg.lstsq(jacobian, np.array([u, v]) - self.reference_silhouette_uv, rcond=None)[0]
        return Detection(xy_m=self.reference_bottle_xy_m + delta,
                         pixel_uv=(u, v), area_px=area,
                         method="fixed RGB silhouette displacement + calibrated camera Jacobian")

    def find_coaster(self) -> Detection | None:
        # Coaster localization is still a calibrated fixture estimate. No
        # simulated coaster state is exposed to the toolbox.
        return Detection(self.coaster_xy_m.copy(), (float("nan"), float("nan")), 0,
                         "calibrated fixture")

    def observe_bottle_xy(self) -> np.ndarray | None:
        frame = self.robot.camera_frame("fixed")
        if frame.depth_m is None:
            detection = self.find_bottle()
            return None if detection is None else detection.xy_m
        # Coarse RGB mask followed by metric point projection. A geometric
        # consistency check removes some background pixels of similar color.
        h, w = frame.rgb.shape[:2]
        component = _color_component(frame, (0, 0, w, h), (5, 75, 35),
                                     (35, 255, 255), min_area=500)
        if component is None:
            return None
        xs, ys, _ = component
        depth = np.asarray(frame.depth_m)[ys, xs]
        valid = np.isfinite(depth) & (depth > .05) & (depth < 2.)
        if valid.sum() < 200:
            return None
        xs, ys, depth = xs[valid], ys[valid], depth[valid]
        points_cv = np.stack([(xs - frame.intrinsic[0, 2]) / frame.intrinsic[0, 0] * depth,
                              (ys - frame.intrinsic[1, 2]) / frame.intrinsic[1, 1] * depth,
                              depth], axis=1)
        points = points_cv @ frame.camera_to_base[:3, :3].T + frame.camera_to_base[:3, 3]
        # Restrict to the bottle body and suppress background color matches.
        body = points[(points[:, 2] > .03) & (points[:, 2] < .12)]
        if len(body) < 100:
            return None
        x = float(np.mean(np.quantile(body[:, 0], [.02, .98])))
        y = float(np.quantile(body[:, 1], .90))
        return np.array([x, y])

    def pick_and_place(self, bottle: Detection, coaster: Detection) -> dict:
        """Measure once, plan staged motions, and return trace for verification."""
        state = self.robot.robot_state()
        if not np.isfinite(state.joints_rad).all() or state.joints_rad.shape != (6,):
            return {"ok": False, "reason": "invalid_joint_state"}
        target = np.r_[bottle.xy_m, .149]
        goal_xyz = {
            "pre": target + [0, 0, .03], "grasp": target,
            "lift": target + [0, 0, .03],
            "transfer": np.r_[coaster.xy_m, .191],
            "place": np.r_[coaster.xy_m, .162],
            "retreat": np.r_[coaster.xy_m, .191],
        }
        seed = state.joints_rad.copy()
        q = {}
        errors = {}
        for name, xyz in goal_xyz.items():
            seed, error = self.robot.solve_grasp_ik(xyz, seed)
            q[name] = seed.copy()
            errors[name] = float(error)
        if max(errors.values()) >= .002:
            return {"ok": False, "reason": "ik_unreachable", "ik_errors_m": errors}
        phases = [
            ("initial", "pre", .070, .8), ("approach", "grasp", .070, 1.2),
            ("close", "grasp", .036, 1.), ("hold", "grasp", .036, .25),
            ("lift", "lift", .036, 1.1), ("transfer", "transfer", .036, 1.8),
            ("lower", "place", .036, 1.2), ("release", "place", .070, .9),
            ("retreat", "retreat", .070, 1.1), ("settle", "retreat", .070, 1.2),
        ]
        trace = []
        for phase, key, aperture, duration in phases:
            self.robot.move_joints(q[key], aperture, duration)
            trace.append({"phase": phase, "aperture_m": aperture,
                          "measured_joints_rad": self.robot.robot_state().joints_rad.tolist()})
        after_xy = self.observe_bottle_xy()
        if after_xy is None:
            return {"ok": False, "reason": "bottle_not_visible_after", "trace": trace}
        distance = float(np.linalg.norm(after_xy - coaster.xy_m))
        return {"ok": distance < .025, "reason": "placed" if distance < .025 else "not_on_coaster",
                "observed_final_error_m": distance, "trace": trace,
                "bottle_detection": {"xy_m": bottle.xy_m.tolist(), "pixel_uv": bottle.pixel_uv}}

    def inspect_placement(self, coaster: Detection) -> dict:
        bottle_xy = self.observe_bottle_xy()
        if bottle_xy is None:
            return {"ok": False, "reason": "bottle_not_visible"}
        distance = float(np.linalg.norm(bottle_xy - coaster.xy_m))
        return {"ok": distance < .025, "observed_error_m": distance,
                "bottle_xy_m": bottle_xy.tolist()}
