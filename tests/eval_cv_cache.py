"""Score a parse cache (e.g. the 5-fold parse_cache_v3_cv.json) the way the GATE uses it, per fold and sub-type.

Run: python tests/eval_cv_cache.py [parse_cache.json] [path/to/SimuHome]

Per request: the gate's preprocessing (clock-time grounding, split-chain repair, device-name repair against the
episode's real device list, chain-anchor repair), device finish times from the gold labels (at runtime they come
from the simulator), then Z3. Reports:
  verdicts           possible/impossible decided correctly
  goals checked ok   a CORRECT schedule for the benchmark goal passes the time check
  goals refused      a CORRECT schedule would be refused (harm; should be ~0)
  goals unchecked    no predicted goal produces the benchmark goal's values, so (G4) the call is not time-checked
"Before" = v2.1 behaviour (no chain-anchor repair; any goal on the device is time-checked).
"""

from __future__ import annotations

import collections
import copy
import json
import os
import re
import sys

import _paths


def main(cache_path: str, simuhome_dir: str) -> int:
    simuhome_dir = _paths.setup(simuhome_dir)
    from irlib.constraint_encoder import check_feasibility, goal_windows
    from irlib.extract_ir_from_episode import extract_dataset
    from irlib.grounding import ground_ir, repair_chain_anchor, repair_device_ids, repair_split_chain, restore_stated_clock
    from irlib.ir_schema import ir_from_compact

    bench = os.path.join(simuhome_dir, "data", "benchmark")
    cache = json.load(open(cache_path, encoding="utf-8"))
    rows = extract_dataset(bench)
    stats = {m: collections.defaultdict(collections.Counter) for m in ("before", "after")}
    problems = collections.defaultdict(list)

    for row in rows:
        entry = cache.get(row["query"])
        if not entry:
            continue
        fold = entry.get("fold", 0)
        qt, case = row["meta"]["query_type"], row["meta"]["case"]
        gold = row["debug_ground_truth"]
        ep = json.load(open(os.path.join(bench, row["source_file"]), encoding="utf-8"))
        known = sorted(set(re.findall(r'"device_id":\s*"([^"]+)"', json.dumps(ep["initial_home_config"]))))
        for mode in ("before", "after"):
            st = stats[mode][(fold, qt)]
            st["n"] += 1
            if "ir" not in entry:
                st["invalid"] += 1
                continue
            if gold["conflict_type"] == "completion_vs_pause":
                continue
            pred = copy.deepcopy(entry["ir"])
            try:
                pred, _ = ground_ir(pred, row["query"])
                pred, _ = repair_split_chain(pred, row["query"])
                pred, _, _ = repair_device_ids(pred, known)
                if mode == "after":
                    pred, _ = restore_stated_clock(pred, row["query"])
                    pred, _ = repair_chain_anchor(pred)
                finish = {r["anchor"]["device_id"]: r.get("resolved_at_minutes") for g in gold["goals"]
                          for r in g["readings"] if r["anchor"].get("kind") == "device_event"}
                for g in pred["goals"]:
                    for r in g["readings"]:
                        if (r.get("anchor") or {}).get("kind") == "device_event":
                            r["resolved_at_minutes"] = finish.get(r["anchor"].get("device_id"))
                ir = ir_from_compact(pred, row["query"], gold["base_time"])
                verdict = check_feasibility(ir).feasible
            except Exception as e:
                st["bad_structure"] += 1
                continue
            want = case == "feasible"
            st[f"{case}_n"] += 1
            st[f"{case}_ok"] += verdict == want
            if not (want and verdict):
                continue
            w = goal_windows(ir)
            tick = float(ep["initial_home_config"]["tick_interval"])
            for gi, g in enumerate(ep["eval"]["goals"]):
                if g["when"].get("relation") == "before" and int(g["when"]["tolerance_ticks"]) == 0:
                    continue  # QT4-3 "still running 1 min before X finishes": a state check, not a schedule time
                st["goals"] += 1
                t = g["when"]["at_tick"] * tick / 60.0
                # the gate (F3) matches a call by the values it CHANGES; "power on" for a device that an earlier
                # goal already switched on changes nothing, so it is only counted when it is the goal's only value
                want_vals = {(x["device_id"], a["attribute"], json.dumps(a["value"])) for x in g["targets"] for a in x["asserts"]
                             if not (a["attribute"].endswith("OnOff.OnOff") and len(x["asserts"]) > 1)}
                devs = {x["device_id"] for x in g["targets"]}
                on_dev = [pg for pg in ir.goals if any(tt.device_id in devs for tt in pg.targets)]
                if mode == "after":  # G4: only goals whose values the correct call would produce are checked
                    cands = [pg for pg in on_dev if any((tt.device_id, a.attribute, json.dumps(a.value)) in want_vals
                                                        for tt in pg.targets for a in tt.asserts)]
                else:
                    cands = on_dev
                if not cands:
                    st["goals_unchecked"] += 1
                elif any(w.get(pg.goal_id) is None or w[pg.goal_id].contains(t, 0.25) for pg in cands):
                    st["goals_ok"] += 1
                else:
                    st["goals_refused"] += 1
                    problems[mode].append((fold, row["source_file"], gi))

    for mode in ("before", "after"):
        print(f"\n=== {mode} ({'v2.1-style' if mode == 'before' else 'v2.2: chain-anchor repair + G4'})")
        tot = collections.Counter()
        for key in sorted(stats[mode]):
            tot.update(stats[mode][key])
        by_qt = collections.defaultdict(collections.Counter)
        for (fold, qt), c in stats[mode].items():
            by_qt[qt].update(c)
        for qt, c in sorted(by_qt.items()):
            print(f"  {qt}: possible verdicts {c['feasible_ok']}/{c['feasible_n']}, impossible verdicts "
                  f"{c['infeasible_ok']}/{c['infeasible_n']}, goals ok {c['goals_ok']}, REFUSED {c['goals_refused']}, "
                  f"unchecked {c['goals_unchecked']} of {c['goals']}, invalid {c['invalid']}")
        print(f"  ALL : possible {tot['feasible_ok']}/{tot['feasible_n']}, impossible {tot['infeasible_ok']}/"
              f"{tot['infeasible_n']}, goals REFUSED {tot['goals_refused']} of {tot['goals']}, unchecked {tot['goals_unchecked']}")
        print("  refused:", problems[mode])
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0] if a else os.path.join(_paths.PROJECT, "data", "parser", "parse_cache_v3_cv.json"),
                  a[1] if len(a) > 1 else os.path.join(_paths.PROJECT, "..", "SimuHome")))
