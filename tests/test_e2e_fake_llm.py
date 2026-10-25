"""End-to-end test of the harness + parallel runner, with a scripted fake LLM.

Run: python tests/test_e2e_fake_llm.py [path/to/SimuHome]

Starts real SimuHome simulator servers and runs SimuHome's own `run_single_config`
(ReAct agent loop, QT4 scorer, judge panel), two episodes in parallel. Only the LLM
is fake: a script that plays the agent, so no API key or GPU is needed. Checks the
behaviour the paper's arms depend on:

  v1_gate     (approach 1): the agent retries a contradictory request and is
              refused every time (the loop seen in the session-1 traces);
  pre_hint    the pre-flight note reaches the agent, which finishes at step 1;
  full_answer the gate answers without a single agent LLM call;
  full        a wrong start time is refused with the window, an invented command
              is refused by the dry run, and the corrected calls go through.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

import _paths


class FakeLLM:
    """Plays the ReAct agent from a script: policy(query, observations) -> (action, input)."""

    def __init__(self, policy, counter, lock):
        self.policy, self.counter, self.lock = policy, counter, lock
        self.model = "fake/scripted-agent"

    def generate(self, messages, response_format=None):
        from irlib.parallel_runner import llm_wait
        with llm_wait():  # as the notebook's shim does around every real LLM request (clock held meanwhile)
            time.sleep(0.2)
        with self.lock:
            self.counter["agent"] += 1
        # The real task is the user turn after SimuHome's one-shot examples; tool results
        # come back as later user turns prefixed "observation:".
        task_idx = max(i for i, m in enumerate(messages) if m.role == "user" and "This is your actual task" in m.content)
        task = messages[task_idx].content
        observations = [m.content for m in messages[task_idx + 1:]
                        if m.role == "user" and m.content.lstrip().lower().startswith("observation")]
        action, args = self.policy(task, observations)
        return json.dumps({"thought": "scripted", "action": action, "action_input": json.dumps(args)})


class FakeJudge:
    model = "fake/judge"

    def __init__(self, counter, lock):
        self.counter, self.lock = counter, lock

    def generate(self, messages, response_format=None):
        with self.lock:
            self.counter["judge"] += 1
        return "A"


def at(base, minutes):
    from datetime import datetime, timedelta
    return (datetime.strptime(base, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def cmd(device, cluster, command, args=None):
    return {"tool": "execute_command", "args": {"device_id": device, "endpoint_id": 1, "cluster_id": cluster,
                                                 "command_id": command, "args": args or {}}}


def main(simuhome_dir: str) -> int:
    simuhome_dir = _paths.setup(simuhome_dir)
    os.chdir(simuhome_dir)
    from irlib.extract_ir_from_episode import extract_pair
    from irlib.harness import Harness
    from irlib.parallel_runner import episode_paths, run_episodes, start_simulators, stop_simulators
    from irlib.shadow_sim import dry_run

    bench = os.path.join(simuhome_dir, "data", "benchmark")
    gold = {}  # perfect-parser stand-in: query -> gold compact IR
    for name in ["qt4-1_infeasible_seed_15", "qt4-1_infeasible_seed_20", "qt4-1_feasible_seed_1", "qt4-1_feasible_seed_2"]:
        ep = json.load(open(os.path.join(bench, name + ".json"), encoding="utf-8"))
        gold[ep["query"]] = extract_pair(ep)["ir_for_training"]
    harness = Harness(parse_fn=lambda q: gold[q.split("\n\n[Note")[0]], dry_run_fn=dry_run, ir_version=3)
    harness.install()

    counter, lock = {"agent": 0, "judge": 0}, threading.Lock()
    out_root = tempfile.mkdtemp(prefix="approach2_e2e_")
    servers = start_simulators(2, simuhome_dir, pausable=True)  # v2.2: clock held during LLM calls
    problems = []

    def run(arm, policy, names):
        harness.set_arm(arm)
        counter.update(agent=0, judge=0)
        n_before = len(harness.telemetry)
        paths = [p for n in names for p in episode_paths(bench, *n)]
        res = run_episodes(paths, servers=servers, run_id=f"e2e_{arm}", model="fake/scripted-agent",
                           make_main_llm=lambda: FakeLLM(policy, counter, lock),
                           make_judge_llms=lambda: [FakeJudge(counter, lock) for _ in range(3)],
                           output_root=out_root, on_episode_start=harness.begin_episode, log=lambda s: None)
        tel = {t["episode"].split("/")[-1]: t for t in harness.telemetry[n_before:]}
        scores = {}
        for p in paths:
            rec = json.load(open(os.path.join(out_root, f"e2e_{arm}", "fake_scripted-agent", p.name), encoding="utf-8"))
            scores[p.stem] = (rec.get("evaluation_result") or {}).get("score")
        return res, dict(counter), tel, scores

    try:
        # ---- v1_gate: the agent keeps retrying a contradictory request
        def retry_policy(query, obs):
            m = re.search(r"At (\d+):(\d+) (AM|PM)", query)
            n = len(obs)
            if n < 3:
                return "schedule_workflow", {"start_time": f"2025-08-23 12:{23 + n}:00",
                                             "steps": [cmd("living_room_on_off_light_1", "OnOff", "On")]}
            return "finish", {"answer": "There is a timing contradiction in the request."}
        res, cnt, tel, scores = run("v1_gate", retry_policy, [("qt4-1", "infeasible", [15, 20])])
        v1_rej = [t["unsat_rejections"] for t in tel.values()]
        print(f"v1_gate:     {res}  agent LLM calls={cnt['agent']}  unsat rejections per episode={v1_rej}  scores={scores}")
        if v1_rej != [3, 3] or cnt["agent"] != 8:
            problems.append(f"v1_gate: expected 3 rejections and 4 agent calls per episode, got {v1_rej}, {cnt}")

        # ---- pre_hint: the note reaches the agent, which finishes at once
        def hint_policy(query, obs):
            if "[Note from the scheduling verifier" in query:
                return "finish", {"answer": "Your request contradicts itself; which time do you want?"}
            return "schedule_workflow", {"start_time": "2025-08-23 12:30:00", "steps": [cmd("x", "OnOff", "On")]}
        res, cnt, tel, scores = run("pre_hint", hint_policy, [("qt4-1", "infeasible", [15, 20])])
        print(f"pre_hint:    {res}  agent LLM calls={cnt['agent']}  preflight={[t['preflight'] for t in tel.values()]}  scores={scores}")
        if cnt["agent"] != 2 or any(t["preflight"] != "unsat" for t in tel.values()):
            problems.append(f"pre_hint: expected 1 agent call per episode, got {cnt}")

        # ---- full_answer: no agent call at all
        res, cnt, tel, scores = run("full_answer", hint_policy, [("qt4-1", "infeasible", [15, 20])])
        print(f"full_answer: {res}  agent LLM calls={cnt['agent']}  judge calls={cnt['judge']}  "
              f"answered_by_gate={[t['answered_by_gate'] for t in tel.values()]}  scores={scores}")
        if cnt["agent"] != 0 or not all(t["answered_by_gate"] for t in tel.values()) or set(scores.values()) != {1}:
            problems.append(f"full_answer: expected 0 agent calls and score 1, got {cnt}, {scores}")

        # ---- full on a feasible episode: wrong time, then invented command, then fixes
        ep = json.load(open(os.path.join(bench, "qt4-1_feasible_seed_1.json"), encoding="utf-8"))
        base = ep["initial_home_config"]["base_time"]
        light, util = "living_room_dimmable_light_1", "utility_room_on_off_light_1"
        script = [
            ("get_room_devices", {"room_id": "living_room"}),
            ("get_room_devices", {"room_id": "utility_room"}),
            ("schedule_workflow", {"start_time": at(base, 18), "steps": [cmd(light, "OnOff", "On"), cmd(light, "LevelControl", "MoveToLevelWithOnOff", {"Level": 30})]}),
            ("schedule_workflow", {"start_time": at(base, 13), "steps": [cmd(light, "OnOff", "On"), cmd(light, "LevelControl", "MoveToLevelWithOnOff", {"Level": 30})]}),
            ("schedule_workflow", {"start_time": at(base, 21), "steps": [cmd(light, "LevelControl", "SetLevel", {"level": 20})]}),
            ("schedule_workflow", {"start_time": at(base, 21), "steps": [cmd(light, "LevelControl", "MoveToLevelWithOnOff", {"Level": 20})]}),
            ("schedule_workflow", {"start_time": at(base, 29), "steps": [cmd(util, "OnOff", "On")]}),
            ("finish", {"answer": "All three actions are scheduled."}),
        ]
        codes = []

        def feasible_policy(query, obs):
            if obs:
                m = re.search(r'"code":\s*(\d+)', obs[-1])
                codes.append(int(m.group(1)) if m else None)
            return script[min(len(obs), len(script) - 1)]
        res, cnt, tel, scores = run("full", feasible_policy, [("qt4-1", "feasible", [1])])
        t = list(tel.values())[0]
        sched_codes = codes[2:7]
        print(f"full:        {res}  agent LLM calls={cnt['agent']}  schedule codes={sched_codes}  "
              f"time_rej={t['time_rejections']} dry_rej={t['dry_run_rejections']} passed={t['passed']}  scores={scores}")
        if sched_codes != [409, 200, 409, 200, 200] or t["time_rejections"] != 1 or t["dry_run_rejections"] != 1:
            problems.append(f"full: expected codes [409, 200, 409, 200, 200], got {sched_codes}; telemetry {t}")
        from irlib.parallel_runner import CLOCK
        print("clock holds:", CLOCK)
        if CLOCK["holds"] == 0 or CLOCK["errors"]:
            problems.append(f"clock hold did not work: {CLOCK}")
        if scores.get("qt4-1_feasible_seed_1") != 1:
            problems.append(f"full: the corrected plan should score 1 with SimuHome's scorer, got {scores}")
    finally:
        stop_simulators(servers)
        shutil.rmtree(out_root, ignore_errors=True)

    if problems:
        print("\nproblems:")
        for p in problems:
            print(" -", p)
    print("\nPASS" if not problems else "\nFAIL")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(_paths.PROJECT, "..", "SimuHome")))
