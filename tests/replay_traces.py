"""Offline check of the current gate version against REAL agent behaviour from an E1 session.

Run: python tests/replay_traces.py <results dir> [path/to/SimuHome] [parse_cache.json]

For every POSSIBLE-request episode in the results (arms baseline, v1_gate, full), rebuild the
episode's home in-process, replay the agent's actions in order (immediate commands are applied,
`schedule_workflow` calls go through the gate, `full` configuration, real parser output from the
parse cache), and record what the gate would answer.

What matters:
  - refusals in episodes SimuHome scored as SUCCESS: correct schedules the gate would block
    (false refusals). Should be 0.
  - refusals in FAILED episodes: mistakes the gate would have flagged.
The replay cannot show what the agent would have done next, so it measures the gate's decisions,
not end-to-end success (that needs a new E1 run).
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys
import time

import _paths


def main(results_dir: str, simuhome_dir: str, cache_path: str) -> int:
    simuhome_dir = _paths.setup(simuhome_dir)
    from irlib.gate import ARMS, EpisodeGate, GateConfig
    from irlib.shadow_sim import dry_run
    from src.pipelines.episode_generation.core.home_augmenter import _initialize_home_locally

    cache = {k: v for k, v in json.load(open(cache_path, encoding="utf-8")).items() if k != "_meta"}
    cfg = GateConfig(**{**ARMS["full"].__dict__, "ir_version": 3})

    def ok(data):
        return {"status": {"code": 200, "message": "OK"}, "data": data, "error": None}

    files = sorted(glob.glob(os.path.join(results_dir, "SimuHome", "experiments", "a2_*_feasible", "*", "qt4-*.json")))
    files = [f for f in files if any(f"a2_{a}_" in f for a in ("baseline", "v1_gate", "full", "selfcheck"))]
    tally = collections.Counter()
    rows = []
    t0 = time.time()
    for i, f in enumerate(files):
        rec = json.load(open(f, encoding="utf-8"))
        score = (rec.get("evaluation_result") or {}).get("score")
        if score not in (0, 1):
            continue
        ep = json.load(open(os.path.join(simuhome_dir, "data", "benchmark", os.path.basename(f)), encoding="utf-8"))
        entry = cache.get(ep["query"])
        if not entry or "ir" not in entry:
            continue
        home, _ = _initialize_home_locally(ep["initial_home_config"])

        def call_tool(name, args, home=home):
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

        def real_schedule(args, home=home):
            r = home.schedule_workflow(args["start_time"], args["steps"])
            return ok(r.data) if r.success else {"status": {"code": 400}, "data": None, "error": {"detail": r.error_message}}

        gate = EpisodeGate(ep["query"], ep["initial_home_config"]["base_time"], entry["ir"], cfg, call_tool, dry_run_fn=dry_run)
        decisions = []
        for st in rec.get("steps") or []:
            act, args = st.get("action"), st.get("action_input") or {}
            if not isinstance(args, dict):
                continue
            try:
                if act == "execute_command":
                    home._execute_command(args["device_id"], args["endpoint_id"], args["cluster_id"], args["command_id"],
                                          args.get("args") or {})
                elif act == "write_attribute":
                    home._write_attribute(args["device_id"], args["endpoint_id"], args["cluster_id"],
                                          args["attribute_id"], args["value"])
                elif act == "schedule_workflow":
                    r = gate.on_schedule(args, real_schedule)
                    code = (r.get("status") or {}).get("code")
                    kind = (r.get("error") or {}).get("type") if code == 409 else ("ok" if code == 200 else f"http{code}")
                    decisions.append(kind)
            except Exception as e:  # a malformed agent call: SimuHome would reject it too
                decisions.append(f"replay_error:{type(e).__name__}")
        refusals = [d for d in decisions if d not in ("ok",) and not d.startswith(("http", "replay_error"))]
        arm = os.path.basename(os.path.dirname(os.path.dirname(f)))
        details = [((st.get("action_input") or {}).get("start_time"), d) for st, d in zip(
            [st for st in rec.get("steps") or [] if st.get("action") == "schedule_workflow"
             and isinstance(st.get("action_input"), dict)], decisions)]
        rows.append(dict(arm=arm, episode=os.path.basename(f)[:-5], score=score, preflight=gate.stats.preflight,
                         decisions=decisions, calls=details))
        tally["episodes"] += 1
        tally[f"calls"] += len(decisions)
        tally[f"refusals_in_{'success' if score else 'failed'}_episodes"] += len(refusals)
        tally[f"episodes_with_refusal_{'success' if score else 'failed'}"] += bool(refusals)
        tally[f"{'success' if score else 'failed'}_episodes"] += 1
        # a refusal the agent later got past (a later call was accepted) vs one with nothing accepted after it
        last_ok = max((k for k, d in enumerate(decisions) if d == "ok"), default=-1)
        blocking = [d for k, d in enumerate(decisions) if d in refusals and k > last_ok]
        tally[f"blocking_refusals_in_{'success' if score else 'failed'}_episodes"] += len(blocking)
        for d in refusals:
            tally[f"kind:{d}|{'success' if score else 'failed'}"] += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(files)} episodes replayed ({time.time() - t0:.0f} s)", flush=True)

    print(json.dumps(dict(sorted(tally.items())), indent=1))
    print("replay errors (agent calls SimuHome would reject anyway):",
          sum(d.startswith("replay_error") for r in rows for d in r["decisions"]))
    print("\nSuccessful episodes with a refusal. A refusal followed by an accepted call = the agent got past it;")
    print("a refusal with nothing accepted after it = the gate would block a schedule SimuHome accepted:")
    for r in rows:
        if r["score"] == 1 and any(d not in ("ok",) and not d.startswith(("http", "replay_error")) for d in r["decisions"]):
            print("  ", r["arm"], r["episode"], r["preflight"], r["decisions"])
    out = os.path.join(results_dir, os.environ.get("REPLAY_OUT", "replay_gate_v22.json"))
    json.dump({"tally": tally, "rows": rows}, open(out, "w"), indent=1)
    print("saved", out)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0], a[1] if len(a) > 1 else os.path.join(_paths.PROJECT, "..", "SimuHome"),
                  a[2] if len(a) > 2 else os.path.join(_paths.PROJECT, "data", "parser", "parse_cache_v3_cv.json")))
