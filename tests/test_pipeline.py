"""Sanity check: v3 extractor + joint Z3 encoder against real SimuHome QT4 episodes.

Run: python tests/test_pipeline.py [path/to/SimuHome/data/benchmark]

CPU-only, no GPU/API keys needed. Two checks:
  1. Verdicts: the encoder, given the ground-truth IR, must call every feasible
     episode feasible and every infeasible one infeasible (duration-model
     conflicts excepted). Approach 1 scored 286/286; v3 must not regress.
  2. Windows: on feasible episodes, the time the benchmark itself checks each goal
     at (`eval.goals[].when.at_tick`) must fall inside the window the encoder
     computes for that goal. A goal outside its window would make the gate reject
     a correct schedule, so this must be 100%.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constraint_encoder import check_feasibility, goal_windows
from src.extract_ir_from_episode import extract_episode, extract_pair
from src.ir_schema import ir_from_compact, uses_goal_anchors


def main(bench_dir: str) -> int:
    paths = sorted(glob.glob(os.path.join(bench_dir, "qt4-*.json")))
    assert paths, f"no qt4-*.json files found under {bench_dir}"

    counts = {"feasible_correct": 0, "feasible_wrong": 0, "infeasible_correct": 0,
              "infeasible_wrong": 0, "infeasible_skipped_duration_model": 0}
    win = {"goals": 0, "inside": 0, "bounded": 0}
    chained_episodes = 0
    wrong: list[str] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            episode = json.load(f)
        ir = extract_episode(episode, for_training=False)
        result = check_feasibility(ir)
        expected_feasible = episode["meta"]["case"] == "feasible"
        if uses_goal_anchors(extract_pair(episode)["ir_for_training"]):
            chained_episodes += 1

        skipped = any("duration model" in (g.reason or "") for g in result.goal_results)
        if not expected_feasible and skipped:
            counts["infeasible_skipped_duration_model"] += 1
            continue
        if result.feasible == expected_feasible:
            counts["feasible_correct" if expected_feasible else "infeasible_correct"] += 1
        else:
            counts["feasible_wrong" if expected_feasible else "infeasible_wrong"] += 1
            wrong.append(f"{os.path.basename(path)}: expected_feasible={expected_feasible} "
                         f"explanation={result.explanation()!r}")

        if expected_feasible:
            windows = goal_windows(ir)
            for gi, raw_goal in enumerate(episode["eval"]["goals"]):
                win["goals"] += 1
                w = windows[gi]
                if w.lo is not None or w.hi is not None:
                    win["bounded"] += 1
                # at_tick, not at_minutes: SimuHome rounds at_minutes to whole minutes.
                at = float(raw_goal["when"]["at_tick"]) * float(episode["initial_home_config"]["tick_interval"]) / 60.0
                if w.contains(at, slack=1e-6):
                    win["inside"] += 1
                else:
                    wrong.append(f"{os.path.basename(path)} goal {gi}: at_minutes={at} outside window [{w.lo}, {w.hi}]")

    print(json.dumps(counts, indent=2))
    checked = sum(counts.values()) - counts["infeasible_skipped_duration_model"]
    correct = counts["feasible_correct"] + counts["infeasible_correct"]
    print(f"Verdict accuracy (excluding duration-model-only conflicts): {correct}/{checked}")
    print(f"Windows: {win['inside']}/{win['goals']} benchmark goal times inside their window "
          f"({win['bounded']} bounded)")
    print(f"Episodes with chained (goal-anchored) readings in the v3 labels: {chained_episodes}")

    # The compact (training) IR must round-trip into the same verdict once anchor times are added.
    rt_bad = 0
    for path in paths:
        episode = json.load(open(path, encoding="utf-8"))
        pair = extract_pair(episode)
        if pair["meta"].get("conflict_type") == "completion_vs_pause":
            continue
        compact = pair["ir_for_training"]
        gold = pair["debug_ground_truth"]
        for gi, g in enumerate(compact["goals"]):
            for ri, r in enumerate(g["readings"]):
                if r["anchor"]["kind"] == "device_event":
                    r["resolved_at_minutes"] = gold["goals"][gi]["readings"][ri].get("resolved_at_minutes")
        verdict = check_feasibility(ir_from_compact(compact, pair["query"], gold["base_time"])).feasible
        rt_bad += verdict != (pair["meta"]["case"] == "feasible")
    print(f"Compact-IR round trip: {rt_bad} verdict mismatches")

    if wrong:
        print(f"\n{len(wrong)} problems (first 15):")
        for line in wrong[:15]:
            print(" -", line)
    ok = not wrong and rt_bad == 0
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    bench_dir = sys.argv[1] if len(sys.argv) > 1 else "../SimuHome/data/benchmark"
    sys.exit(main(bench_dir))
