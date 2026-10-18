"""Exploratory Piper bottle Sim2Sim demo; no robot or neural-policy execution."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene/scene.xml"
BASE_B = np.array([.2882762818, .1067295579, .149])
BASE_D = np.array([.2816166465, -.0712457258, .161])
PHASES = [
    ("initial", "pre", .035, .8), ("approach", "grasp", .035, 1.2),
    ("close", "grasp", .018, 1.0), ("hold", "grasp", .018, .25),
    ("lift", "lift", .018, 1.1), ("transfer", "transfer", .018, 1.8),
    ("lower", "place", .018, 1.2), ("release", "place", .035, .9),
    ("retreat", "retreat", .035, 1.1), ("settle", "retreat", .035, 1.2),
]


def solve(m, d, pads, target, seed, qa):
    def residual(q):
        d.qpos[qa] = np.r_[q, .035, -.035]
        mujoco.mj_forward(m, d)
        rotation = d.geom_xmat[pads[0]].reshape(3, 3)
        return np.r_[10 * (d.geom_xpos[pads].mean(0) - target),
                     rotation[:, 1] - [0, 0, -1], .0001 * (q - seed)]
    result = least_squares(residual, seed, bounds=(m.jnt_range[:6, 0], m.jnt_range[:6, 1]), max_nfev=400)
    error = float(np.linalg.norm(result.fun[:3]) / 10)
    return result.x, error


def run_case(case, skill, model_xy, out):
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 9)])
    va = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 9)])
    bq = int(m.jnt_qposadr[m.joint("bottle_free").id])
    bv = int(m.jnt_dofadr[m.joint("bottle_free").id])
    pads = [m.geom("pad7").id, m.geom("pad8").id]
    fingers = [m.geom("finger_collision7").id, m.geom("finger_collision8").id]
    bottle = m.geom("bottle_proxy").id
    coaster = m.geom("white_pad").id
    # Hidden target environment: these parameters are applied only to execution.
    true_xy = BASE_B[:2] + np.array(case["bottle_delta_xy_m"])
    if "bottle_radius_scale" in case:
        m.geom_size[bottle, 0] *= float(case["bottle_radius_scale"])
    if "bottle_friction" in case:
        m.geom_friction[bottle, 0] = float(case["bottle_friction"])
    target_xy = true_xy if skill == "skill_only" else model_xy if skill == "adaptive" else BASE_B[:2]
    target_b = np.r_[target_xy, BASE_B[2]]
    seed = np.array([.33552273199, 1.78587811714, -.92826397005, 0, .80038163225, 1.50475268])
    goals = {}
    errors = []
    for name, target in [("pre", target_b + [0, 0, .03]), ("grasp", target_b),
                         ("lift", target_b + [0, 0, .03]),
                         ("transfer", BASE_D + [0, 0, .03]),
                         ("place", BASE_D + [0, 0, .001]),
                         ("retreat", BASE_D + [0, 0, .03])]:
        seed, error = solve(m, d, pads, target, seed, qa)
        goals[name] = seed.copy()
        errors.append(error)
    mujoco.mj_resetData(m, d)
    # Shared reachable home: nominal pre-grasp pose with an open gripper.
    home, home_error = solve(m, d, pads, BASE_B + [0, 0, .03], seed, qa)
    errors.append(home_error)
    d.qpos[qa] = np.r_[home, .035, -.035]
    d.qpos[bq:bq + 2] = true_xy
    mujoco.mj_forward(m, d)
    rows = []
    prev = d.qpos[qa].copy()
    for phase, key, gap, duration in PHASES:
        goal = np.r_[goals[key], gap, -gap]
        steps = round(duration / m.opt.timestep)
        for k in range(steps):
            a = (k + 1) / steps
            a = a ** 3 * (10 - 15 * a + 6 * a * a)
            d.ctrl[:] = (1 - a) * prev + a * goal
            d.qfrc_applied[:] = 0
            d.qfrc_applied[va[:6]] = d.qfrc_bias[va[:6]]
            mujoco.mj_step(m, d)
            force = np.zeros(2)
            support = 0.0
            penetration = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                penetration = max(penetration, -float(c.dist))
                pair = {c.geom1, c.geom2}
                if bottle in pair and (coaster in pair or any(f in pair for f in fingers)):
                    f = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, f)
                    if coaster in pair:
                        support += f[0]
                    for j, finger in enumerate(fingers):
                        if finger in pair:
                            force[j] += f[0]
            rows.append([d.time, *d.qpos[bq:bq + 3],
                         np.linalg.norm(d.qvel[bv:bv + 3]), *force,
                         support, penetration, phase])
        prev = goal
    a = np.asarray([r[:9] for r in rows], dtype=float)
    phases = np.asarray([r[9] for r in rows])
    carry = np.isin(phases, ["lift", "transfer"])
    hold = np.minimum(a[carry, 5], a[carry, 6]) > .1
    result = {
        "case": case["name"], "skill": skill,
        "hidden_bottle_xy_m": true_xy.tolist(), "model_bottle_xy_m": np.asarray(model_xy).tolist(),
        "hidden_bottle_friction": float(m.geom_friction[bottle, 0]),
        "hidden_bottle_radius_m": float(m.geom_size[bottle, 0]),
        "ik_max_error_m": max(errors), "final_xyz_m": a[-1, 1:4].tolist(),
        "xy_error_m": float(np.linalg.norm(a[-1, 1:3] - BASE_D[:2])),
        "max_lift_m": float(np.max(a[:, 3] + .015)),
        "carry_bilateral_fraction": float(np.mean(hold)),
        "max_penetration_m": float(a[:, 8].max()),
        "final_support_N": float(a[-500:, 7].mean()),
        "final_speed_m_s": float(a[-500:, 4].max()),
        "warnings": d.warning.number.tolist(),
    }
    result["success"] = bool(result["ik_max_error_m"] < .002 and
        result["xy_error_m"] < .015 and abs(a[-1, 3] + .003) < .003 and
        result["max_lift_m"] > .02 and result["carry_bilateral_fraction"] > .99 and
        result["max_penetration_m"] < .0015 and result["final_support_N"] > 1 and
        result["final_speed_m_s"] < .01 and not any(result["warnings"]))
    np.savez_compressed(out / f"{case['name']}_{skill}.npz", metrics=a, phase=phases)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "cases.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # Frozen before running: development counterexample and unseen validation cases.
    cases = json.loads(args.cases.read_text())
    (args.out / "cases.json").write_text(json.dumps(cases, indent=2))
    results = []
    for case in cases:
        # Diagnostic factorial controls. The true XY supplied to skill_only
        # is an oracle pose observation, so it cannot prove learned perception.
        # adaptive uses the same pose as a calibrated model estimate.
        for skill in ("fixed", "model_only", "skill_only", "adaptive"):
            model_xy = BASE_B[:2] if skill in ("fixed", "skill_only") else BASE_B[:2] + case["bottle_delta_xy_m"]
            result = run_case(case, skill, model_xy, args.out)
            results.append(result)
            (args.out / "results.json").write_text(json.dumps(results, indent=2))
            print(case["name"], skill, result["success"],
                  "xy_error", round(result["xy_error_m"], 4), flush=True)


if __name__ == "__main__":
    main()
