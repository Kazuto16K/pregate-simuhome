"""E1 results from SimuHome's per-episode records: accuracy, cost, and PAIRED comparisons between arms.

Reads `<experiments>/<run_id>/<model>/<episode>.json` (SimuHome's own output format), so it works on one
Kaggle session's folder, on several sessions' zips unpacked into one folder (shards merge by file union), and
locally. Run ids look like `a2_<arm>_<qt>_<case>[_ch][_gXY]`; `_ch` = clock held during LLM calls, `_gXY` =
gate version (gate arms only).

Paired comparison: every arm runs on the same episodes, so arm A vs arm B is compared episode by episode
(McNemar's exact test on the discordant pairs). With n = 50 per cell this detects a difference of about
8-10 episodes; the unpaired 8-10 per cell of the first sessions could not.

No SimuHome import; standard library only.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Optional

RUN_RE = re.compile(r"^a2_(?P<arm>.+?)_(?P<qt>qt4_[123])_(?P<case>feasible|infeasible)(?P<tags>(?:_[a-z0-9]+)*)$")


def parse_run_id(run_id: str) -> Optional[dict]:
    m = RUN_RE.match(run_id)
    if not m:
        return None
    tags = [t for t in m.group("tags").split("_") if t]
    return {"arm": m.group("arm"), "qt": m.group("qt").replace("_", "-"), "case": m.group("case"),
            "clock_hold": "ch" in tags, "gate": next((t[1:] for t in tags if re.fullmatch(r"g\d+", t)), None)}


def load_records(experiments_dir: str, require_tags: tuple[str, ...] = ()) -> list[dict]:
    """One row per scored episode: run info + seed + score + cost + per-goal results."""
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(experiments_dir, "a2_*"))):
        run_id = os.path.basename(run_dir)
        info = parse_run_id(run_id)
        if info is None or any(f"_{t}" not in run_id for t in require_tags):
            continue
        for f in glob.glob(os.path.join(run_dir, "*", "qt4-*.json")):
            try:
                r = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            ev = r.get("evaluation_result") or {}
            score = ev.get("score")
            steps = r.get("steps") or []
            goals = ev.get("goals_eval") or []
            rows.append({
                **info, "run_id": run_id, "seed": int(r.get("seed") or os.path.basename(f).split("_seed_")[1][:-5]),
                "score": score if score in (0, 1) else None, "infra_error": score not in (0, 1),
                "agent_calls": len(r.get("raw_responses") or []), "seconds": float(r.get("duration") or 0.0),
                "schedule_calls": sum(1 for s in steps if s.get("action") == "schedule_workflow"),
                "goals": len(goals), "goals_met": sum(1 for g in goals if g.get("satisfied")),
            })
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b = (A right, B wrong), c = (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summary_table(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[(r["qt"], r["case"], r["arm"])].append(r)
    out = []
    for (qt, case, arm), xs in sorted(by.items()):
        ok = [x for x in xs if x["score"] is not None]
        n = max(1, len(ok))
        goals = sum(x["goals"] for x in ok)
        out.append({
            "qt": qt, "case": case, "arm": arm, "evaluated": len(ok), "correct": sum(x["score"] for x in ok),
            "accuracy": round(sum(x["score"] for x in ok) / n, 3), "infra_errors": sum(x["infra_error"] for x in xs),
            "goals_met": f"{sum(x['goals_met'] for x in ok)}/{goals}" if case == "feasible" else "",
            "agent_calls_per_episode": round(sum(x["agent_calls"] for x in ok) / n, 2),
            "seconds_per_episode": round(sum(x["seconds"] for x in ok) / n, 1),
            "schedule_calls_per_episode": round(sum(x["schedule_calls"] for x in ok) / n, 2),
        })
    return out


def paired(rows: list[dict], arm_a: str, arm_b: str) -> list[dict]:
    """arm_a vs arm_b on the episodes both scored, per (qt, case) and pooled per case."""
    score = {(r["qt"], r["case"], r["arm"], r["seed"]): r["score"] for r in rows if r["score"] is not None}
    out, pooled = [], defaultdict(lambda: {"n": 0, "a": 0, "b": 0, "only_a": 0, "only_b": 0})
    for qt, case in sorted({(r["qt"], r["case"]) for r in rows}):
        seeds = sorted({k[3] for k in score if k[0] == qt and k[1] == case})
        both = [s for s in seeds if (qt, case, arm_a, s) in score and (qt, case, arm_b, s) in score]
        if not both:
            continue
        a = [score[(qt, case, arm_a, s)] for s in both]
        b = [score[(qt, case, arm_b, s)] for s in both]
        c = {"n": len(both), "a": sum(a), "b": sum(b),
             "only_a": sum(x == 1 and y == 0 for x, y in zip(a, b)), "only_b": sum(x == 0 and y == 1 for x, y in zip(a, b))}
        for k, v in c.items():
            pooled[case][k] += v
        out.append((qt, case, c))
    out += [("all", case, c) for case, c in sorted(pooled.items())]
    return [{"qt": qt, "case": case, "n": c["n"], arm_a: c["a"], arm_b: c["b"], f"only_{arm_a}": c["only_a"],
             f"only_{arm_b}": c["only_b"], "p_mcnemar": round(mcnemar_exact(c["only_a"], c["only_b"]), 4)}
            for qt, case, c in out]


def print_report(rows: list[dict], gate_arm: str = "full", others: tuple[str, ...] = ("baseline", "selfcheck")) -> None:
    print(f"{'sub-type':8s} {'case':10s} {'arm':12s} correct/n   acc   goals met   calls/ep  s/ep   sched/ep infra")
    for s in summary_table(rows):
        print(f"{s['qt']:8s} {s['case']:10s} {s['arm']:12s} {s['correct']:>3}/{s['evaluated']:<4}  {s['accuracy']:.2f}  "
              f"{s['goals_met']:>9s}  {s['agent_calls_per_episode']:>7}  {s['seconds_per_episode']:>6}  "
              f"{s['schedule_calls_per_episode']:>6}  {s['infra_errors']:>3}")
    arms = {r["arm"] for r in rows}
    for other in others:
        for g in (gate_arm, f"{gate_arm}_answer"):
            if g in arms and other in arms:
                print(f"\npaired: {g} vs {other} (same episodes; only_X = episodes only X got right)")
                for p in paired(rows, g, other):
                    print("  ", p)


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "SimuHome/experiments"
    print_report(load_records(d, tuple(sys.argv[2:])))
