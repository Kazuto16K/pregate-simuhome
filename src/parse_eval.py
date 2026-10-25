"""Score a parse cache the way the gate uses it (shared by the rewording-robustness notebook and the tests).

For every benchmark request: the gate's preprocessing (clock-time grounding, split-chain repair, stated-clock
restoration, device-name repair against the episode's real devices, chain-anchor repair), device finish times
from the gold labels (at runtime they come from the simulator), then Z3. `text_for` maps a benchmark query to the
text the parser actually saw (e.g. a reworded version); grounding is applied to THAT text, as at runtime.

Counted per (sub-type, case):
  verdict ok            possible judged possible / contradiction proven
  goals ok / refused    a correct schedule for a benchmark goal passes / would be refused by the time check
  goals unchecked       no predicted goal produces the goal's values, so the gate does not time-check it
  invalid               no usable parse (the gate lets the episode through unchecked)
QT4-3 "still running before X finishes" goals are state checks, not schedule times, and are not counted.
"""

from __future__ import annotations

import collections
import copy
import json
import os
import re
from typing import Callable, Optional

from .constraint_encoder import check_feasibility, goal_windows
from .grounding import ground_ir, repair_chain_anchor, repair_device_ids, repair_split_chain, restore_stated_clock
from .ir_schema import ir_from_compact


def gate_view(pred: dict, text: str, known_devices: list[str]) -> dict:
    """The parse after the gate's deterministic preprocessing (same order as EpisodeGate.ensure_verdict)."""
    pred = copy.deepcopy(pred)
    pred, _ = ground_ir(pred, text)
    pred, _ = repair_split_chain(pred, text)
    pred, _ = restore_stated_clock(pred, text)
    if known_devices:
        pred, _, _ = repair_device_ids(pred, known_devices)
    pred, _ = repair_chain_anchor(pred)
    return pred


def score(cache: dict, rows: list[dict], bench_dir: str, text_for: Optional[Callable[[str], Optional[str]]] = None):
    """Returns (stats by (query_type, case), list of problems). `cache` maps the text the parser saw -> {'ir': ...}."""
    text_for = text_for or (lambda q: q)
    stats = collections.defaultdict(collections.Counter)
    problems = []
    for row in rows:
        qt, case = row["meta"]["query_type"], row["meta"]["case"]
        gold = row["debug_ground_truth"]
        if gold["conflict_type"] == "completion_vs_pause":
            continue
        text = text_for(row["query"])
        if text is None:
            continue  # no valid rewording for this request: excluded, counted by the caller
        st = stats[(qt, case)]
        st["n"] += 1
        entry = cache.get(text)
        if not entry or "ir" not in entry:
            st["invalid"] += 1
            problems.append((row["source_file"], "invalid parse"))
            continue
        ep = json.load(open(os.path.join(bench_dir, row["source_file"]), encoding="utf-8"))
        known = sorted(set(re.findall(r'"device_id":\s*"([^"]+)"', json.dumps(ep["initial_home_config"]))))
        try:
            pred = gate_view(entry["ir"], text, known)
            finish = {r["anchor"]["device_id"]: r.get("resolved_at_minutes") for g in gold["goals"] for r in g["readings"]
                      if r["anchor"].get("kind") == "device_event"}
            for g in pred["goals"]:
                for r in g["readings"]:
                    if (r.get("anchor") or {}).get("kind") == "device_event":
                        r["resolved_at_minutes"] = finish.get(r["anchor"].get("device_id"))
            ir = ir_from_compact(pred, text, gold["base_time"])
            verdict = check_feasibility(ir).feasible
        except Exception as e:
            st["invalid"] += 1
            problems.append((row["source_file"], f"bad structure: {type(e).__name__}"))
            continue
        want = case == "feasible"
        st["verdict_ok"] += verdict == want
        if verdict != want:
            problems.append((row["source_file"], "wrong verdict: predicted " + ("possible" if verdict else "impossible")))
        if not (want and verdict):
            continue
        w = goal_windows(ir)
        tick = float(ep["initial_home_config"]["tick_interval"])
        for gi, g in enumerate(ep["eval"]["goals"]):
            if g["when"].get("relation") == "before" and int(g["when"]["tolerance_ticks"]) == 0:
                continue
            st["goals"] += 1
            t = g["when"]["at_tick"] * tick / 60.0
            want_vals = {(x["device_id"], a["attribute"], json.dumps(a["value"])) for x in g["targets"] for a in x["asserts"]
                         if not (a["attribute"].endswith("OnOff.OnOff") and len(x["asserts"]) > 1)}
            devs = {x["device_id"] for x in g["targets"]}
            cands = [pg for pg in ir.goals if any(tt.device_id in devs for tt in pg.targets)
                     and any((tt.device_id, a.attribute, json.dumps(a.value)) in want_vals for tt in pg.targets for a in tt.asserts)]
            if not cands:
                st["goals_unchecked"] += 1
            elif any(w.get(pg.goal_id) is None or w[pg.goal_id].contains(t, 0.25) for pg in cands):
                st["goals_ok"] += 1
            else:
                st["goals_refused"] += 1
                problems.append((row["source_file"], f"goal {gi}: a correct schedule would be refused"))
    return stats, problems


def table(stats: dict) -> list[dict]:
    """One row per sub-type: possible judged possible, contradictions proven, goals refused, invalid."""
    out = []
    for qt in sorted({k[0] for k in stats}):
        f, i = stats.get((qt, "feasible"), collections.Counter()), stats.get((qt, "infeasible"), collections.Counter())
        out.append({"qt": qt, "possible_ok": f"{f['verdict_ok']}/{f['n']}", "contradictions_proven": f"{i['verdict_ok']}/{i['n']}",
                    "goals_refused": f"{f['goals_refused']}/{f['goals']}", "goals_unchecked": f["goals_unchecked"],
                    "invalid": f["invalid"] + i["invalid"]})
    return out
