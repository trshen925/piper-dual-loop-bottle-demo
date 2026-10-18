# Bottle-on-coaster skill

This file documents the fixed `RobotAPI` and the editable `BottleToolbox` for a coding agent writing `agent_script.py`.

## Task

Find the yellow bottle and white coaster, grasp the bottle with a continuous 36 mm aperture, lift, transfer, place, release, and check the observed result. The example task script uses `find_bottle()`, `find_coaster()`, and `pick_and_place()`.

## Fixed low-level API

`camera_frame("fixed" | "wrist")` returns RGB, optional aligned metric depth, OpenCV intrinsics, timestamp, and `T_base_camera` in metres. `robot_state()` returns measured J1–J6 radians and continuous aperture metres. `solve_grasp_ik(target_xyz_m, seed_rad)` returns a 6-joint plan and position error. `move_joints(joints_rad, aperture_m, duration_s)` commands absolute targets. The API is implemented by `MuJoCoRobot` for simulation; a real adapter must implement the same interface.

Toolbox code must not read MuJoCo `qpos` for object state, contacts, or hidden case parameters. The simulator harness may use them only for reset and scoring.

## Measured limits

- Current RGB detector is a **local calibration** for the fixed camera and yellow bottle. It uses a connected color component and a nominal image-to-world Jacobian. On the four frozen sim cases, initial XY errors were 0–1.5 mm. It has not been validated on real camera images or changed lighting.
- `find_coaster()` currently returns a calibrated fixture coordinate. If the coaster moves, replace it with camera-based localization before executing.
- The program checks the bottle after placement with RGBD; on frozen cases, visual error is about 6–7 mm. A visual placement check alone cannot detect the x+8 mm penetration failure.
- The x+8 mm case produced 3.14 mm maximum penetration, above the 1.5 mm physical threshold, even though the task script reported placement. Preserve this as a regression case.
- The tool currently plans all phases once and does not react to a slipping bottle during transfer. Do not treat a reported `placed` result as sufficient for real execution.

## Real adapter contract

Use the calibrated PiPER camera order: fixed `camera2_fixed`, wrist `camera1_wrist`. For the real robot, supply the measured joint state rather than the last command, transform wrist extrinsics with the current link6 pose, convert SDK aperture to the continuous `joint7-joint8` convention, and use bounded joint motion with an external stop. The repository does not ship a Piper SDK implementation or a verified real perception calibration. Do not command hardware until that adapter and its calibration are tested separately.

## SimEx-style iteration

1. Probe the current toolbox on varied bottle/coaster poses and contact settings.
2. Select one systematic failure and freeze a focused task set.
3. Edit `toolbox.py` and this file together; replay solved tasks to catch regressions.
4. For a real rollout, save raw images, joint/action traces, and task outcome. Replay the **same fixed script** in sim, align relevant behavior, screen several distinct toolbox repairs, then select the next candidate.

The current repository supplies the execution boundary and one small probe set. It does not implement autonomous agent-driven autoresearch or real rollout replay.
