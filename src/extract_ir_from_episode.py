"""Component: deterministic (query, IR) pair extraction from SimuHome episodes.

No LLM calls, no manual labeling. SimuHome benchmark/generated episodes
already carry, in `eval.goals` (ground-truth required behavior) and
`temporal_conflict` (ground-truth contradiction, infeasible episodes only),
every field needed to derive the IR mechanically. See handoff Sec 5.3.

Two IRs are produced per episode:
  - `ir_for_training`: what the SLM must learn to produce from the query
    text alone. Device-anchored readings never carry a numeric resolved
    time here, because the query text does not state one — the live
    simulator state does, and only the runtime agent can fetch it.
  - `debug_ground_truth`: the same IR with anchor times filled in from the
    simulator-computed fields, for exercising the Z3 encoder end-to-end
    without needing a live simulator.

IR v3 (approach 2) differs from v2 in two labelling rules, both read off
`eval.goals[].when`:
  - a goal with no device anchor but `reference_minutes > 0` is chained to the
    earlier goal whose `at_minutes` equals that reference ("8 minutes after the
    previous action") -> Anchor(kind="goal", goal_ref=j);
  - `relation == "before"` with `tolerance_ticks == 0` is a deadline (the state
    must hold by then, e.g. QT4-3 "start the dryer now") -> relation "by".
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Optional

from .ir_schema import Anchor, Assertion, Goal, IntentIR, Target, TimeReading, compact_ir_dict

# Conflict types where the contradiction is a stated-time mismatch (checkable
# from two TimeReadings alone). See handoff Sec 6: duration/resource
# feasibility conflicts (e.g. "finish washer exactly when dishwasher ends"
# but the washer cycle itself is longer) need a device duration model the
# Z3 encoder does not have yet — flagged, not silently mis-scored.
_DURATION_MODEL_CONFLICT_TYPES = {"completion_vs_pause"}


def _tolerance_minutes(when: dict[str, Any], tick_interval: float) -> float:
    return float(when.get("tolerance_ticks", 0)) * tick_interval / 60.0


def _chained_goal_ref(
    goal_id: int, when: dict[str, Any], raw_goals: list[dict[str, Any]], *, any_direction: bool = False
) -> Optional[int]:
    """Index of the goal this one is measured from, or None.

    SimuHome writes "8 minutes after the previous action" as relation "after",
    offset 8, with `reference_minutes` = the previous goal's `at_minutes` and no
    device anchor; the latest earlier goal with that time is the reference.
    Deadline goals (QT4-3, `any_direction=True`) reference the OTHER goal's time
    ("the dryer must be running 1 min before it is paused"), which can be a later goal.
    """
    ref = when.get("reference_minutes")
    if not ref:
        return None
    for j in range(goal_id - 1, -1, -1):
        if raw_goals[j]["when"].get("at_minutes") == ref:
            return j
    if any_direction:
        for j in range(goal_id + 1, len(raw_goals)):
            if raw_goals[j]["when"].get("at_minutes") == ref:
                return j
    return None


def _device_anchor(raw_anchor: dict[str, Any]) -> Anchor:
    return Anchor(
        kind="device_event",
        room_id=raw_anchor.get("room_id"),
        device_id=raw_anchor.get("device_id"),
        device_type=raw_anchor.get("device_type"),
    )


def _goal_from_eval_goal(
    goal_id: int,
    raw_goal: dict[str, Any],
    tick_interval: float,
    raw_goals: Optional[list[dict[str, Any]]] = None,
    for_training: bool = True,
) -> Goal:
    when = raw_goal["when"]
    raw_anchor = raw_goal.get("anchor") or {}

    relation = when.get("relation", "after")
    deadline = relation == "before" and int(when.get("tolerance_ticks", 0)) == 0
    if deadline:
        relation = "by"  # deadline: the state must hold no later than anchor - offset

    goal_ref = _chained_goal_ref(goal_id, when, raw_goals or [], any_direction=deadline)
    if deadline and goal_ref is not None:
        # Measured from the other goal, even when eval.goals labels a device anchor:
        # its reference_minutes is that goal's time, not the device's finish.
        anchor = Anchor(kind="goal", goal_ref=goal_ref)
    elif raw_anchor.get("device_id"):
        anchor = _device_anchor(raw_anchor)
    elif goal_ref is not None:
        anchor = Anchor(kind="goal", goal_ref=goal_ref)
    else:
        anchor = Anchor(kind="now")

    reading = TimeReading(
        relation=relation,
        anchor=anchor,
        offset_minutes=float(when.get("offset_minutes", 0.0)),
        tolerance_minutes=_tolerance_minutes(when, tick_interval),
    )
    if anchor.kind == "device_event" and not for_training and when.get("reference_tick") is not None:
        # Debug IR only: when the anchor device finishes, as the runtime reads it from
        # the simulator (CountdownTime). The training IR never contains it. Ticks, not
        # reference_minutes, which SimuHome rounds to whole minutes.
        reading.resolved_at_minutes = float(when["reference_tick"]) * tick_interval / 60.0

    targets = [
        Target(
            room_id=t["room_id"],
            device_id=t["device_id"],
            device_type=t.get("device_type"),
            asserts=[
                Assertion(
                    attribute=a["attribute"],
                    value=a["value"],
                    value_type=a.get("value_type"),
                    description=a.get("description"),
                )
                for a in t.get("asserts", [])
            ],
        )
        for t in raw_goal.get("targets", [])
    ]

    return Goal(goal_id=goal_id, readings=[reading], targets=targets)


def _inject_conflicting_reading(
    goal: Goal, temporal_conflict: dict[str, Any], for_training: bool, raw_anchor: Optional[dict[str, Any]] = None
) -> Goal:
    """Add the second, contradicting TimeReading to the primary goal.

    Mutates and returns `goal`. If the conflict is duration/resource-based
    (no comparable second time reading exists), flags the goal instead.
    """
    conflict_type = temporal_conflict.get("type")

    if conflict_type in _DURATION_MODEL_CONFLICT_TYPES or "conflict_at" not in temporal_conflict:
        goal.requires_duration_model = True
        return goal

    anchor_end_at = temporal_conflict.get("anchor_end_at")
    expected_at = temporal_conflict.get("expected_at")
    conflict_time = temporal_conflict.get("conflict_time")
    relation = temporal_conflict.get("relation", "after")

    if anchor_end_at is not None:
        # Dependency-anchored primary reading (e.g. "20 min after dishwasher finishes").
        base_offset = temporal_conflict.get(
            "offset_minutes", temporal_conflict.get("delay_minutes", 0.0)
        )
        offset = float(base_offset)
        # The conflict is stated against the anchor DEVICE finishing, so use the
        # device from eval.goals even if v3 relabelled the goal as goal-anchored.
        anchor = _device_anchor(raw_anchor) if (raw_anchor or {}).get("device_id") else goal.readings[0].anchor
        primary_reading = TimeReading(
            relation=relation,
            anchor=anchor,
            offset_minutes=offset,
            tolerance_minutes=goal.readings[0].tolerance_minutes,
            # resolved_at_minutes here is the ANCHOR EVENT's own resolved
            # time (e.g. when the dishwasher finishes), not the target time
            # — the Z3 encoder derives target = anchor +/- offset itself.
            resolved_at_minutes=None if for_training else float(anchor_end_at),
        )
        goal.readings[0] = primary_reading
    elif expected_at is not None:
        # Now-anchored primary reading (e.g. "13 minutes from now").
        goal.readings[0].offset_minutes = float(expected_at)
        goal.readings[0].anchor = Anchor(kind="now")

    if conflict_time is not None:
        goal.readings.append(
            TimeReading(
                relation="at",
                anchor=Anchor(kind="now"),
                absolute_time=conflict_time.split(" ")[-1] if " " in conflict_time else conflict_time,
                tolerance_minutes=goal.readings[0].tolerance_minutes,
            )
        )

    return goal


def extract_episode(episode: dict[str, Any], *, for_training: bool = True) -> IntentIR:
    """Build an IntentIR from one raw SimuHome benchmark/generated episode dict."""
    meta = episode["meta"]
    tick_interval = float(episode["initial_home_config"]["tick_interval"])
    base_time = episode["initial_home_config"]["base_time"]
    case = meta.get("case", "unknown")

    raw_goals = episode["eval"]["goals"]
    goals = [
        _goal_from_eval_goal(i, rg, tick_interval, raw_goals, for_training) for i, rg in enumerate(raw_goals)
    ]

    conflict_type = None
    if case == "infeasible" and "temporal_conflict" in episode:
        temporal_conflict = episode["temporal_conflict"]
        conflict_type = temporal_conflict.get("type")
        first = goals[0].readings[0] if goals and goals[0].readings else None
        if (
            first is not None
            and first.relation == "by"
            and first.anchor.kind == "goal"
            and conflict_type not in _DURATION_MODEL_CONFLICT_TYPES
            and temporal_conflict.get("conflict_time")
        ):
            # QT4-3: every goal keeps exactly the reading its FEASIBLE twin has; the stated
            # clock time is added to the goal it is about. v3.0 instead replaced goal 0's
            # reading, so feasible and infeasible QT4-3 labels differed in more than the clock
            # time, and the parser copied the infeasible template (with an invented clock time)
            # onto feasible requests (held-out seeds 55, 75).
            #   absolute_time_mismatch: "start at 6:14 PM ... pause when X finishes" -> the start (goal 0)
            #   delay_time_mismatch:    "start 14 min after X finishes, at 11:40"  -> the delayed start
            #   impossible_early_end:   "finish together before 2:35 PM"           -> the finish
            if conflict_type == "absolute_time_mismatch":
                target = goals[0]
            else:
                target = next((g for g in goals if g.readings and g.readings[0].anchor.kind == "device_event"), goals[0])
            clock = temporal_conflict["conflict_time"]
            target.readings.append(
                TimeReading(
                    relation="at",
                    anchor=Anchor(kind="now"),
                    absolute_time=clock.split(" ")[-1],
                    tolerance_minutes=target.readings[0].tolerance_minutes,
                )
            )
        elif goals:
            goals[0] = _inject_conflicting_reading(
                goals[0], temporal_conflict, for_training, raw_goals[0].get("anchor")
            )

    return IntentIR(
        query=episode["query"],
        query_type=meta.get("query_type", "unknown"),
        case=case,
        base_time=base_time,
        goals=goals,
        conflict_type=conflict_type,
    )


def extract_pair(episode: dict[str, Any]) -> dict[str, Any]:
    """Return {"query", "ir_for_training", "debug_ground_truth", "meta"} for one episode."""
    training_ir = extract_episode(episode, for_training=True)
    debug_ir = extract_episode(episode, for_training=False)
    return {
        "query": episode["query"],
        "ir_for_training": compact_ir_dict(training_ir.to_dict()),
        "debug_ground_truth": debug_ir.to_dict(),
        "meta": episode["meta"],
    }


def extract_dataset(benchmark_dir: str, pattern: str = "qt4-*_seed_*.json") -> list[dict[str, Any]]:
    """Extract (query, IR) pairs for every QT4 episode file in `benchmark_dir`."""
    # Benchmark files sit directly in the dir; generator output is <run>/episodes/*.json.
    paths = sorted(
        glob.glob(os.path.join(benchmark_dir, "qt4-*.json"))
        + glob.glob(os.path.join(benchmark_dir, "*", "episodes", "qt4-*.json"))
    )
    pairs = []
    errors = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                episode = json.load(f)
            pair = extract_pair(episode)
            pair["source_file"] = os.path.basename(path)
            pairs.append(pair)
        except Exception as e:  # keep going; report at the end
            errors.append((path, str(e)))
    if errors:
        print(f"[extract_dataset] {len(errors)} file(s) failed to extract:")
        for path, err in errors[:20]:
            print(f"  {os.path.basename(path)}: {err}")
    return pairs


if __name__ == "__main__":
    import sys

    bench_dir = sys.argv[1] if len(sys.argv) > 1 else "SimuHome/data/benchmark"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/qt4_ir_pairs.jsonl"
    pairs = extract_dataset(bench_dir)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Extracted {len(pairs)} pairs -> {out_path}")
