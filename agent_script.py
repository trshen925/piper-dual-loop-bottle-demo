"""Short task script analogous to the SimEx website's agent_script.py."""


def run_task(tb, max_attempts=1):
    for attempt in range(max_attempts):
        bottle = tb.find_bottle()
        coaster = tb.find_coaster()
        if bottle is None or coaster is None:
            return {"ok": False, "reason": "perception_failed", "attempt": attempt}
        result = tb.pick_and_place(bottle, coaster)
        result["attempt"] = attempt
        if result["ok"] or result["reason"] in {"ik_unreachable", "invalid_joint_state"}:
            return result
    return result
