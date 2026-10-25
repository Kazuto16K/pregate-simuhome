"""Regression tests for gate v2.1 (F3, F4), on the two QT4-3 episodes that failed in E1 session 1.

Run: python tests/test_fixes_v21.py [path/to/SimuHome]

Real SimuHome homes in-process, gold IR as the parser, the agent's actions replayed by hand:
  seed 45: the agent pressed Start before powering the washer on (start silently fails), then
           scheduled the pause. v2.0 only attached an ignored note. v2.1 must refuse with
           NO_EFFECT (F4); the correct sequence (power on, then start) must pass.
  seed 30: "pause dryer 1 when dryer 2 finishes". SimuHome checks "dryer 1 still running" exactly 1 min
           before dryer 2 finishes (tolerance 0) and "paused" within +/-30 s of the finish. Checked in a
           simulated home: a pause 1.1 min early fails the first check, 0.9 min early passes both. So
           v2.2 (G1) must refuse -1.1 and pass -0.9 and 0; a pause after the window must be matched to the
           pause goal (F3) and refused. (v2.1's test wrongly assumed -1.1 passes.)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

import _paths


def main(simuhome_dir: str) -> int:
    simuhome_dir = _paths.setup(simuhome_dir)
    from irlib.extract_ir_from_episode import extract_pair
    from irlib.gate import ARMS, EpisodeGate, GateConfig
    from irlib.shadow_sim import dry_run
    from src.pipelines.episode_generation.core.home_augmenter import _initialize_home_locally

    def ok(data):
        return {"status": {"code": 200, "message": "OK"}, "data": data, "error": None}

    def tools_for(home):
        def call_tool(name, args):
            if name == "get_current_time":
                return ok({"now": home.get_virtual_now_str()})
            if name == "get_attribute":
                r = home._get_attribute(args["device_id"], args["endpoint_id"], args["cluster_id"], args["attribute_id"])
                return ok(r.data)
            if name == "get_home_state":
                return ok(home._get_home_state().data)
            if name == "get_workflow_status":
                r = home.get_workflow_status(args["workflow_id"])
                return ok(r.data) if r.success else {"status": {"code": 400}, "data": None, "error": {}}
            raise KeyError(name)

        def real_schedule(args):
            r = home.schedule_workflow(args["start_time"], args["steps"])
            return ok(r.data) if r.success else {"status": {"code": 400}, "data": None, "error": {"detail": r.error_message}}

        return call_tool, real_schedule

    def at(base, minutes):
        return (datetime.strptime(base, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")

    def cmd(dev, cluster, command, args=None):
        return {"tool": "execute_command", "args": {"device_id": dev, "endpoint_id": 1, "cluster_id": cluster,
                                                     "command_id": command, "args": args or {}}}

    full = GateConfig(**{**ARMS["full"].__dict__, "ir_version": 3})
    problems = []

    def scenario(seed, prepare, pause_minutes):
        ep = json.load(open(os.path.join(simuhome_dir, "data", "benchmark", f"qt4-3_feasible_seed_{seed}.json"),
                            encoding="utf-8"))
        base = ep["initial_home_config"]["base_time"]
        tick = ep["initial_home_config"]["tick_interval"]
        home, _ = _initialize_home_locally(ep["initial_home_config"])
        prepare(home)
        call_tool, real_schedule = tools_for(home)
        gate = EpisodeGate(ep["query"], base, extract_pair(ep)["ir_for_training"], full, call_tool, dry_run_fn=dry_run)
        gate.ensure_verdict()
        dev = ep["eval"]["goals"][1]["targets"][0]["device_id"]
        gold = ep["eval"]["goals"][1]["when"]["at_tick"] * tick / 60.0
        r = gate.on_schedule({"start_time": at(base, gold + pause_minutes), "steps": [cmd(dev, "OperationalState", "Pause")]},
                             real_schedule)
        return r, gate

    # ---- seed 45: Start before power-on -> the washer never runs -> pausing it does nothing
    washer = "utility_room_laundry_washer_1"

    def wrong_order(home):
        home._execute_command(washer, 1, "OperationalState", "Start")  # fails silently: still off
        home._execute_command(washer, 1, "OnOff", "On")

    def right_order(home):
        home._execute_command(washer, 1, "OnOff", "On")
        home._execute_command(washer, 1, "OperationalState", "Start")

    r_bad, g_bad = scenario(45, wrong_order, 0.0)
    r_good, g_good = scenario(45, right_order, 0.0)
    print("seed 45, washer never started -> pause:", r_bad["status"]["code"], (r_bad.get("error") or {}).get("type"))
    print("   message:", ((r_bad.get("error") or {}).get("detail") or "")[:230])
    print("seed 45, washer running        -> pause:", r_good["status"]["code"], "| effect-matched:", g_good.stats.effect_matched_calls)
    if not (r_bad["status"]["code"] == 409 and r_bad["error"]["type"] == "WORKFLOW_WOULD_CHANGE_NOTHING"):
        problems.append("seed 45: no-effect pause not refused")
    if r_good["status"]["code"] != 200:
        problems.append(f"seed 45: correct pause refused: {r_good}")

    # ---- seed 30: pause ~1 minute early -> must be matched to the pause goal and refused
    dryer = "utility_room_laundry_dryer_1"

    def started(home):
        home._execute_command(dryer, 1, "OnOff", "On")
        home._execute_command(dryer, 1, "OperationalState", "Start")

    r_early, _ = scenario(30, started, -1.1)
    r_slightly, _ = scenario(30, started, -0.9)
    r_ok, g_ok = scenario(30, started, 0.0)
    r_late, g_late = scenario(30, started, +1.5)
    print("seed 30, pause 1.1 min early   ->", r_early["status"]["code"], "(SimuHome fails this: must be refused)")
    print("   message:", ((r_early.get("error") or {}).get("detail") or "")[:260])
    print("seed 30, pause 0.9 min early   ->", r_slightly["status"]["code"], "(SimuHome accepts this: must pass)")
    print("seed 30, pause on time         ->", r_ok["status"]["code"], "| effect-matched:", g_ok.stats.effect_matched_calls)
    print("seed 30, pause 1.5 min late    ->", r_late["status"]["code"], (r_late.get("error") or {}).get("type"))
    print("   message:", ((r_late.get("error") or {}).get("detail") or "")[:260])
    if r_early["status"]["code"] != 409:
        problems.append(f"seed 30: pause before the 'still running' check not refused: {r_early}")
    if r_slightly["status"]["code"] != 200:
        problems.append(f"seed 30: pause after the 'still running' check refused (the scorer accepts it): {r_slightly}")
    if r_ok["status"]["code"] != 200:
        problems.append(f"seed 30: on-time pause refused: {r_ok}")
    # too late: dryer 1 may already have finished by then, so either the time check or the
    # no-effect check (pausing a stopped dryer) may refuse it; both are correct
    if r_late["status"]["code"] != 409:
        problems.append("seed 30: late pause not refused")

    for p in problems:
        print(" -", p)
    print("\nPASS" if not problems else "\nFAIL")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(_paths.PROJECT, "..", "SimuHome")))
