"""Integration test: the approach-2 gate against real SimuHome homes (no LLM, no GPU).

Run: python tests/test_gate.py [path/to/SimuHome]

Each QT4 benchmark episode's home is built in-process with SimuHome's own
initializer; the gate's tool calls go straight to that Home object. The gold
compact IR (what a perfect v3 parser would output) stands in for the parser.

Checks, over all 300 QT4 episodes:
  1. pre-flight verdict == benchmark label, with device anchors read LIVE from the
     simulator (CountdownTime), not from the answer key;
  2. feasible: a call at the benchmark's own goal time is never refused (0 false
     time rejections); a call 5 minutes after the last goal on that device is refused
     whenever the window is bounded (earlier calls may legitimately serve a later goal,
     because SimuHome scores the state at each goal's time);
  3. infeasible: schedule_workflow is refused with the final-rejection message,
     and the gate fails open after max_unsat_rejections;
  4. dry run: a valid light command passes, an invented command is refused with
     SimuHome's error text, and the cap on dry-run rejections holds.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timedelta

import _paths


def main(simuhome_dir: str) -> int:
    simuhome_dir = _paths.setup(simuhome_dir)
    from irlib.extract_ir_from_episode import extract_pair
    from irlib.gate import EpisodeGate, GateConfig
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
                return ok(r.data) if r.success else {"status": {"code": 400}, "data": None, "error": {"detail": r.error_message}}
            if name == "get_home_state":
                return ok(home._get_home_state().data)
            if name == "get_workflow_status":
                r = home.get_workflow_status(args["workflow_id"])
                return ok(r.data) if r.success else {"status": {"code": 400}, "data": None, "error": {"detail": r.error_message}}
            raise KeyError(name)

        def real_schedule(args):
            r = home.schedule_workflow(args["start_time"], args["steps"])
            return ok(r.data) if r.success else {"status": {"code": 400}, "data": None, "error": {"detail": r.error_message}}

        return call_tool, real_schedule

    def at(base_time, minutes):
        return (datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")

    paths = sorted(glob.glob(os.path.join(simuhome_dir, "data", "benchmark", "qt4-*.json")))
    assert paths, "no QT4 episodes found"
    c = {"verdict_ok": 0, "verdict_n": 0, "gold_calls": 0, "gold_refused": 0, "off_calls": 0, "off_refused": 0,
         "unsat_refused": 0, "unsat_n": 0, "off_other_goal": 0}
    problems = []
    time_cfg = GateConfig(preflight="hint", time_check="on", dry_run="off", target_check="off", ir_version=3,
                          max_plan_rejections=10**6)

    for path in paths:
        episode = json.load(open(path, encoding="utf-8"))
        pair = extract_pair(episode)
        case = episode["meta"]["case"]
        if episode["meta"].get("conflict_type") == "completion_vs_pause":
            continue
        cfg_home = episode["initial_home_config"]
        base = cfg_home["base_time"]
        tick = float(cfg_home["tick_interval"])
        home, _ = _initialize_home_locally(cfg_home)
        call_tool, real_schedule = tools_for(home)
        gate = EpisodeGate(episode["query"], base, pair["ir_for_training"], time_cfg, call_tool)

        action, _text = gate.preflight_action()
        c["verdict_n"] += 1
        if gate.infeasible == (case == "infeasible") and action == ("hint" if case == "infeasible" else "none"):
            c["verdict_ok"] += 1
        else:
            problems.append(f"{os.path.basename(path)}: preflight={gate.stats.preflight} action={action} case={case}")
            continue

        if case == "infeasible":
            c["unsat_n"] += 1
            dev = pair["ir_for_training"]["goals"][0]["targets"][0]["device_id"]
            call = {"start_time": at(base, 30), "steps": [{"tool": "execute_command", "args": {"device_id": dev}}]}
            r = gate.on_schedule(call, real_schedule)
            if r["status"]["code"] == 409 and r["error"]["type"] == "TEMPORAL_CONTRADICTION_IN_REQUEST":
                c["unsat_refused"] += 1
            else:
                problems.append(f"{os.path.basename(path)}: infeasible call not refused: {r}")
            continue

        for gi, raw_goal in enumerate(episode["eval"]["goals"]):
            gold_min = float(raw_goal["when"]["at_tick"]) * tick / 60.0
            dev = raw_goal["targets"][0]["device_id"]
            steps = [{"tool": "execute_command", "args": {"device_id": dev}}]
            # A fresh gate per call keeps the two probes independent of each other.
            # "off" probe: 5 min after the LAST window on this device, i.e. too late for every goal on it.
            # (SimuHome scores the state at the goal time, so an earlier call may legitimately serve a later
            # goal on the same device; only a call late for all of them is certainly wrong.)
            last_same_dev = max(float(g2["when"]["at_tick"]) * tick / 60.0 for g2 in episode["eval"]["goals"]
                                if any(t["device_id"] == dev for t in g2["targets"]))
            for shift, key in ((0.0, "gold"), (last_same_dev - gold_min + 5.0, "off")):
                g2 = EpisodeGate(episode["query"], base, pair["ir_for_training"], time_cfg, call_tool)
                g2.ensure_verdict()
                w = g2.windows.get(gi)
                if key == "off" and (w is None or w.lo is None or w.hi is None):
                    continue  # unbounded on one side (deadline): a later/earlier call may be fine
                start = gold_min + shift
                if start <= 0.05:
                    continue  # SimuHome refuses start times that are not in the future
                r = g2.on_schedule({"start_time": at(base, start), "steps": steps}, lambda a: ok({"workflow_id": "x"}))
                refused = r["status"]["code"] == 409
                if key == "off" and not refused:
                    # Legitimate when another goal on the same device allows that time
                    # (e.g. two goals on one light, 8 minutes apart).
                    others = [o.goal_id for o in g2.ir.goals if o.goal_id != gi
                              and any(t.device_id == dev for t in o.targets)
                              and g2.windows[o.goal_id].contains(start, time_cfg.slack_minutes)]
                    if others:
                        c["off_other_goal"] += 1
                        continue
                    problems.append(f"{os.path.basename(path)} goal {gi}: +5 min not refused, window [{w.lo}, {w.hi}]")
                c[f"{key}_calls"] += 1
                c[f"{key}_refused"] += refused
                if key == "gold" and refused:
                    problems.append(f"{os.path.basename(path)} goal {gi}: gold time refused: {r['error']['detail'][:160]}")

    # 3b. fail-open cap on an infeasible request
    ep = json.load(open(os.path.join(simuhome_dir, "data", "benchmark", "qt4-1_infeasible_seed_15.json"), encoding="utf-8"))
    home, _ = _initialize_home_locally(ep["initial_home_config"])
    call_tool, real_schedule = tools_for(home)
    cap_cfg = GateConfig(preflight="hint", max_unsat_rejections=2)
    g = EpisodeGate(ep["query"], ep["initial_home_config"]["base_time"], extract_pair(ep)["ir_for_training"], cap_cfg, call_tool)
    call = {"start_time": at(ep["initial_home_config"]["base_time"], 22), "steps": [
        {"tool": "execute_command", "args": {"device_id": "living_room_on_off_light_1", "endpoint_id": 1,
                                              "cluster_id": "OnOff", "command_id": "On", "args": {}}}]}
    codes = [g.on_schedule(call, real_schedule)["status"]["code"] for _ in range(3)]
    cap_ok = codes == [409, 409, 200]
    if not cap_ok:
        problems.append(f"unsat cap: codes {codes}, expected [409, 409, 200]")

    # 4. dry run on a feasible QT4-1 episode
    ep = json.load(open(os.path.join(simuhome_dir, "data", "benchmark", "qt4-1_feasible_seed_1.json"), encoding="utf-8"))
    base = ep["initial_home_config"]["base_time"]
    home, _ = _initialize_home_locally(ep["initial_home_config"])
    call_tool, real_schedule = tools_for(home)
    dr_cfg = GateConfig(preflight="hint", time_check="on", dry_run="reject", target_check="warn", ir_version=3,
                        max_plan_rejections=1)
    g = EpisodeGate(ep["query"], base, extract_pair(ep)["ir_for_training"], dr_cfg, call_tool, dry_run_fn=dry_run)
    light = "living_room_dimmable_light_1"
    good = {"start_time": at(base, 13), "steps": [
        {"tool": "execute_command", "args": {"device_id": light, "endpoint_id": 1, "cluster_id": "OnOff", "command_id": "On", "args": {}}},
        {"tool": "execute_command", "args": {"device_id": light, "endpoint_id": 1, "cluster_id": "LevelControl",
                                              "command_id": "MoveToLevelWithOnOff", "args": {"Level": 30}}}]}
    bad = {"start_time": at(base, 21), "steps": [
        {"tool": "execute_command", "args": {"device_id": light, "endpoint_id": 1, "cluster_id": "LevelControl",
                                              "command_id": "SetLevel", "args": {"level": 20}}}]}
    r_good = g.on_schedule(good, real_schedule)
    r_bad1 = g.on_schedule(bad, real_schedule)
    r_bad2 = g.on_schedule(bad, real_schedule)  # cap = 1: second attempt fails open
    dry_ok = (r_good["status"]["code"] == 200 and r_bad1["status"]["code"] == 409
              and r_bad1["error"]["type"] == "WORKFLOW_STEP_WOULD_FAIL" and "SetLevel" in r_bad1["error"]["detail"]
              and r_bad2["status"]["code"] == 200)
    if not dry_ok:
        problems.append(f"dry run: good={r_good} bad1={r_bad1} bad2={r_bad2}")
    note = (r_good.get("data") or {}).get("verifier_note")

    print(json.dumps(c, indent=2))
    print(f"pre-flight verdicts (live anchors): {c['verdict_ok']}/{c['verdict_n']}")
    print(f"feasible, call at the gold time: {c['gold_refused']}/{c['gold_calls']} refused (must be 0)")
    print(f"feasible, call 5 min after the device's last goal (bounded): {c['off_refused']}/{c['off_calls']} refused "
          f"(+{c['off_other_goal']} that land in another goal's window on the same device, correctly passed)")
    print(f"infeasible, final rejection: {c['unsat_refused']}/{c['unsat_n']}")
    print(f"fail-open cap after 2 rejections: {'ok' if cap_ok else 'WRONG'}")
    print(f"dry run (valid passes, invented command refused, cap holds): {'ok' if dry_ok else 'WRONG'}")
    print(f"   refusal text: {((r_bad1.get('error') or {}).get('detail') or '')[:170]}")
    print(f"   target note on the valid call: {note!r}")
    if problems:
        print(f"\n{len(problems)} problems (first 15):")
        for p in problems[:15]:
            print(" -", p)
    passed = (not problems and c["gold_refused"] == 0 and c["off_refused"] == c["off_calls"]
              and c["unsat_refused"] == c["unsat_n"])
    print("\nPASS" if passed else "\nFAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(_paths.PROJECT, "..", "SimuHome")))
