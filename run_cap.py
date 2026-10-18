"""Run the same short task script against the fixed MuJoCo RobotAPI."""
import argparse
import json
from pathlib import Path

import numpy as np

from agent_script import run_task
from sim_adapter import MuJoCoRobot
from toolbox import BottleToolbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parent / "cases.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    results = []
    for case in cases:
        robot = MuJoCoRobot()
        robot.reset(np.array([.2882762818, .1067295579]) + case["bottle_delta_xy_m"])
        try:
            toolbox = BottleToolbox(robot)
            outcome = run_task(toolbox)
            evaluation = robot.evaluation() if robot.metrics else {}
            if evaluation:
                final_xy = np.asarray(evaluation["final_bottle_xyz_m"][:2])
                physical_error = float(np.linalg.norm(final_xy - toolbox.coaster_xy_m))
                evaluation["final_xy_error_m"] = physical_error
                evaluation["physical_acceptance"] = bool(
                    physical_error < .015 and evaluation["max_lift_m"] > .02 and
                    evaluation["max_penetration_m"] < .0015 and
                    not any(evaluation["warnings"]))
            record = {"case": case["name"], "split": case["split"],
                      "program_result": outcome, "simulator_evaluation": evaluation}
            results.append(record)
            (args.out / "results.json").write_text(json.dumps(results, indent=2))
            print(case["name"], outcome.get("ok"), outcome.get("reason"),
                  evaluation.get("max_penetration_m"), flush=True)
        finally:
            robot.close()


if __name__ == "__main__":
    main()
