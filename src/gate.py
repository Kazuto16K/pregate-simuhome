"""Component 3 (approach 2): the per-episode gate around `schedule_workflow`.

Approach 1 re-parsed the user's request on every `schedule_workflow` call, checked
only the request (never the agent's arguments) and sent back a 409 the agent
could not act on, so it kept retrying with other times. Approach 2 changes this:

  A. Pre-flight: parse once and run Z3 once, before the agent's first step. Device
     anchors are read straight from the simulator (no LLM call). If the request is
     contradictory, either tell the agent at step 0 ("hint") or answer the user
     directly without calling the agent at all ("answer").
  B. Final rejection: when the contradiction is in the request itself, the 409 says
     that no start_time can fix it and that the agent should call finish.
  C. Plan check, for consistent requests:
     C1 the call's start_time must fall inside the window Z3 computes for the goal
        whose devices the call touches;
     C2 the steps are dry-run on a copy of the home at that start_time
        (`shadow_sim`): steps that would fail are rejected with SimuHome's own error
        message; target attributes that would not reach the requested value are
        reported (warn) or rejected.
  Every rejection kind is capped per episode (`max_plan_rejections`); after that the
  gate fails open, so it can slow the agent down but never trap it in a loop.

v2.1 (after E1 session 1, see results/e1_session1_analysis.md):
  F1 explanations in plain words with every device finish time spelled out (explain.py);
  F2 the timer readings behind a verdict are shown: answer mode records them as steps, hint
     mode asks the agent to read the timer itself and quote the finish time;
  F3 a schedule call is matched to the goal whose target state its steps actually produce
     on the copy of the house (a Pause matches the "paused" goal), not the goal nearest in time;
  F4 a workflow that changes nothing on the copy (e.g. pausing a washer that is not running)
     is refused, unless the devices already are in a state the request asks for.

v2.2 (after E1 session 2, see docs/FINDINGS.md section 10):
  G1 a call that would overwrite an earlier DEADLINE goal on the same device is refused if it comes before
     that deadline (QT4-3: "start the dishwasher now, pause it when the dryer finishes" is checked as
     "still running 1 min before the dryer finishes"; agents paused 1-5 min early). v2.1 compared with the
     deadline goal's window START, which for a deadline is "now", so the rule never fired;
  G2 a refused (or let-through) failing step comes with SimuHome's full reason and the cluster's real
     commands and argument names (agents wrote {"level": 60} for MoveToLevel(Level, ...), wrote read-only
     attributes, and never learned why: SimuHome's workflow record keeps only "Invalid arguments");
  G3 the parser's "now + N and previous goal + M" split of "N min after the previous action" is merged
     (grounding.repair_split_chain), and a chain pointing at another device's action is re-pointed to the
     previous action on the SAME device (grounding.repair_chain_anchor; 56 of 56 benchmark labels chain that
     way), and a clock time the user stated but the parser dropped is put back for QT4-3-shaped requests
     (grounding.restore_stated_clock: 17 missed QT4-3 contradictions). All found in the 5-fold parsers'
     held-out errors;
  G4 the time check applies only to a call the dry run MATCHES to a goal (it produces that goal's values).
     A call for a goal the parser missed used to be checked against another goal on the same device (at
     another time) and refused although it was right.

Nothing here imports SimuHome: tool calls, the parser and the dry run are passed in,
so the logic is unit-testable without a simulator or GPU.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .constraint_encoder import (
    FeasibilityResult,
    GoalWindow,
    check_feasibility,
    clock_to_minutes,
    goal_windows,
    minutes_to_clock,
)
from .explain import TimerRead, agent_hint, device_label, refusal_detail, user_reply
from .grounding import ground_ir, repair_chain_anchor, repair_device_ids, repair_split_chain, restore_stated_clock
from .ir_schema import ir_from_compact, uses_goal_anchors
from .shadow_sim import values_match

ToolFn = Callable[[str, dict], dict]


@dataclass
class GateConfig:
    preflight: str = "hint"  # "off" | "hint" | "answer"
    check_at_schedule: bool = True  # enforce the request-level verdict inside schedule_workflow
    terminal_rejection: bool = True  # B: say the contradiction cannot be fixed by another time
    time_check: str = "auto"  # "off" | "on" | "auto" (= on only for v3 IRs, which have goal anchors)
    dry_run: str = "reject"  # "off" | "warn" | "reject": steps that would fail when they run
    target_check: str = "warn"  # "off" | "warn" | "reject": target attributes not reaching the value
    no_effect_check: str = "reject"  # "off" | "reject": F4, refuse workflows that change nothing (needs dry_run)
    sat_hints: bool = False  # also tell the agent the computed times when the request is consistent
    ground_clock_times: bool = True  # drop clock times the parser wrote but the user never said (grounding.py)
    ground_device_names: bool = True  # map device ids that do not exist in the house to the closest real one
    repair_chains: bool = True  # G3: merge "now + N" / "previous goal + 0" split readings (grounding.py)
    max_plan_rejections: int = 2  # per episode and per kind (time / dry_run / target); then fail open
    max_unsat_rejections: int = 3  # per episode, for an infeasible request
    slack_minutes: float = 0.25  # extra slack on every window (SimuHome scores within +/-30 s)
    # No dry run for start times further ahead than this (the copy must be fast-forwarded the whole
    # way: ~1 s per 10 simulated minutes). QT4 goals are at most ~116 min after the request.
    dry_run_horizon_minutes: float = 150.0
    ir_version: int = 2  # 3 when the parser was trained on v3 labels


# The arms of the experiment. `None` = SimuHome's original tool, no gate.
ARMS: dict[str, Optional[GateConfig]] = {
    "baseline": None,
    "selfcheck": None,  # + SELF_CHECK_PROMPT appended to the query (see the notebook)
    # approach 1 reproduced inside the same harness: check at schedule time only
    "v1_gate": GateConfig(preflight="off", terminal_rejection=False, time_check="off", dry_run="off", target_check="off",
                          no_effect_check="off", ground_clock_times=False, ground_device_names=False),
    "pre_hint": GateConfig(preflight="hint", time_check="off", dry_run="off", target_check="off", no_effect_check="off"),
    "pre_answer": GateConfig(preflight="answer", time_check="off", dry_run="off", target_check="off", no_effect_check="off"),
    "full": GateConfig(preflight="hint"),
    "full_answer": GateConfig(preflight="answer"),
}


def _response(code: int, message: str, err_type: str, detail: str) -> dict:
    return {"status": {"code": code, "message": message}, "data": None, "error": {"type": err_type, "detail": detail}}


def _ok(resp: Any) -> bool:
    return isinstance(resp, dict) and (resp.get("status") or {}).get("code") == 200


def _tool_data(resp: Any) -> Any:
    return resp.get("data") if isinstance(resp, dict) else None


def explain_contradiction(result: FeasibilityResult) -> str:
    """Plain-English sentence naming the clashing statements (from the unsat core)."""
    bad = [g for g in result.goal_results if not g.sat]
    parts = [g.reason for g in bad if g.reason]
    return "; ".join(parts) if parts else "two parts of the request give the same action different times"


def user_answer(result: FeasibilityResult) -> str:
    """The reply sent to the user when the gate answers without the agent ("answer" pre-flight)."""
    return (
        "I can't schedule this as requested, because the timing contradicts itself: "
        f"{explain_contradiction(result)}. Both cannot be true, so I have not scheduled anything. "
        "Please tell me which time you want."
    )


def devices_in_steps(steps: list) -> list[str]:
    out = []
    for st in steps or []:
        args = st.get("args") if isinstance(st, dict) else None
        if isinstance(args, dict) and isinstance(args.get("device_id"), str) and args["device_id"] not in out:
            out.append(args["device_id"])
    return out


@dataclass
class GateStats:
    preflight: str = "not_run"  # not_run | sat | unsat | parse_fail | error
    preflight_seconds: float = 0.0
    clock_readings_dropped: int = 0  # parser clock times not found in the request (grounding filter)
    device_names_repaired: int = 0  # parser device ids not in the house, mapped to the closest real device
    device_names_unresolved: int = 0  # ... and ones with no close match (left as they are)
    chains_repaired: int = 0  # G3
    clock_readings_restored: int = 0  # G3: stated clock times the parser left out
    time_unchecked: int = 0  # G4: the call matched no goal by its effect, so no time check
    schedule_calls: int = 0
    passed: int = 0
    unsat_rejections: int = 0
    time_rejections: int = 0
    dry_run_rejections: int = 0
    target_rejections: int = 0
    target_warnings: int = 0
    no_effect_rejections: int = 0  # F4
    effect_matched_calls: int = 0  # F3: calls matched to a goal by what they produce
    unmatched_calls: int = 0  # touched no device the IR knows: not checked
    fail_open: int = 0  # a cap was reached, or a check could not run
    dry_run_seconds: float = 0.0
    events: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: (v if k != "events" else list(v)) for k, v in self.__dict__.items()}


class EpisodeGate:
    """All gate state for ONE episode (one user request)."""

    def __init__(
        self,
        query: str,
        base_time: str,
        compact_ir: Optional[dict],
        cfg: GateConfig,
        call_tool: ToolFn,
        dry_run_fn: Optional[Callable[..., Any]] = None,
    ):
        self.query = query
        self.base_time = base_time  # when the user spoke = minute 0 of every reading
        self.compact_ir = compact_ir
        self.cfg = cfg
        self.call_tool = call_tool
        self.dry_run_fn = dry_run_fn
        self.stats = GateStats()
        self.result: Optional[FeasibilityResult] = None
        self.windows: dict[int, GoalWindow] = {}
        self.ir = None
        self.committed: list[dict] = []
        self._verdict_done = False
        self.timer_reads: dict[str, TimerRead] = {}  # F2: what the verdict is based on
        self.evidence_calls: list[tuple[str, dict, Any]] = []  # (tool, args, response) the gate made

    # ------------------------------------------------------------------ verdict
    def _house_device_ids(self) -> list[str]:
        """Every device id in the house, from the simulator (one call; no answer key involved)."""
        snap = _tool_data(self.call_tool("get_home_state", {})) or {}
        return [d.get("device_id") for room in (snap.get("rooms") or {}).values() for d in (room.get("devices") or [])
                if d.get("device_id")]

    def _now_minutes(self) -> float:
        now = (_tool_data(self.call_tool("get_current_time", {})) or {}).get("now")
        return clock_to_minutes(self.base_time, now) if now else 0.0

    def _resolve_device_anchors(self, compact: dict) -> None:
        """Fill resolved_at_minutes for device anchors from the live simulator:
        finish time = now + CountdownTime. Idle devices stay unresolved (symbolic)."""
        needed = {
            r["anchor"]["device_id"]
            for g in compact["goals"]
            for r in g["readings"]
            if (r.get("anchor") or {}).get("kind") == "device_event" and r["anchor"].get("device_id")
            and r.get("resolved_at_minutes") is None
        }
        if not needed:
            return
        now_min = self._now_minutes()
        finish: dict[str, Optional[float]] = {}
        for dev in needed:
            args = {"device_id": dev, "endpoint_id": 1, "cluster_id": "OperationalState", "attribute_id": "CountdownTime"}
            resp = self.call_tool("get_attribute", args)
            self.evidence_calls.append(("get_attribute", args, resp))
            data = _tool_data(resp)
            value = data.get("value") if isinstance(data, dict) else data
            ok = isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
            finish[dev] = now_min + float(value) / 60.0 if ok else None
            self.timer_reads[dev] = TimerRead(dev, float(value) if ok else None, finish[dev])
        for g in compact["goals"]:
            for r in g["readings"]:
                a = r.get("anchor") or {}
                if a.get("kind") == "device_event" and r.get("resolved_at_minutes") is None:
                    r["resolved_at_minutes"] = finish.get(a.get("device_id"))

    def ensure_verdict(self) -> None:
        if self._verdict_done:
            return
        self._verdict_done = True
        t0 = time.time()
        if self.compact_ir is None:
            self.stats.preflight = "parse_fail"
            return
        try:
            compact = copy.deepcopy(self.compact_ir)
            if self.cfg.ground_clock_times:
                compact, self.stats.clock_readings_dropped = ground_ir(compact, self.query)
            if self.cfg.repair_chains:
                compact, self.stats.chains_repaired = repair_split_chain(compact, self.query)
                compact, self.stats.clock_readings_restored = restore_stated_clock(compact, self.query)
            if self.cfg.ground_device_names:
                known = self._house_device_ids()
                if known:
                    compact, self.stats.device_names_repaired, self.stats.device_names_unresolved =                         repair_device_ids(compact, known)
            if self.cfg.repair_chains:
                compact, n = repair_chain_anchor(compact)
                self.stats.chains_repaired += n
            self._resolve_device_anchors(compact)
            self.ir = ir_from_compact(compact, self.query, self.base_time)
            self.result = check_feasibility(self.ir)
            if self.result.feasible and self._time_check_on():
                self.windows = goal_windows(self.ir)
            self.stats.preflight = "sat" if self.result.feasible else "unsat"
        except Exception as e:  # a bug in the check must never break the episode: fail open
            self.result = None
            self.stats.preflight = "error"
            self.stats.events.append(("verdict_error", repr(e)[:200]))
        finally:
            self.stats.preflight_seconds += time.time() - t0

    def _time_check_on(self) -> bool:
        mode = self.cfg.time_check
        if mode == "auto":
            return self.cfg.ir_version >= 3 or bool(self.compact_ir and uses_goal_anchors(self.compact_ir))
        return mode == "on"

    @property
    def infeasible(self) -> bool:
        return self.result is not None and not self.result.feasible

    # ---------------------------------------------------------------- pre-flight
    def preflight_action(self) -> tuple[str, str]:
        """("none" | "hint" | "answer", text) to apply before the agent's first step."""
        if self.cfg.preflight == "off":
            return "none", ""
        self.ensure_verdict()
        if self.infeasible:
            if self.cfg.preflight == "answer":
                return "answer", user_reply(self.ir, self.result, self.timer_reads)
            return "hint", agent_hint(self.ir, self.result, self.timer_reads)
        if self.result is not None and self.cfg.sat_hints:
            parts = []
            for g in self.result.goal_results:
                if g.resolved_minutes is None:
                    continue
                devs = ", ".join(t.device_id for t in self.ir.goals[g.goal_id].targets) or "?"
                parts.append(f"{devs} at {minutes_to_clock(self.base_time, g.resolved_minutes)}")
            if parts:
                return "hint", "\n\n[Note from the scheduling verifier: the requested times work out to " + "; ".join(parts) + ".]"
        return "none", ""

    # ------------------------------------------------------------ schedule call
    def on_schedule(self, args: dict, real_tool: Callable[[dict], dict]) -> dict:
        self.stats.schedule_calls += 1
        if not self.cfg.check_at_schedule:
            return self._commit(args, real_tool)
        self.ensure_verdict()
        if self.result is None:  # parse failure or check error: fail open
            self.stats.fail_open += 1
            return self._commit(args, real_tool)

        if self.infeasible:
            if self.stats.unsat_rejections >= self.cfg.max_unsat_rejections:
                self.stats.fail_open += 1
                return self._commit(args, real_tool)
            self.stats.unsat_rejections += 1
            if self.cfg.terminal_rejection:
                return _response(409, "REQUEST_CONTRADICTION", "TEMPORAL_CONTRADICTION_IN_REQUEST",
                                 refusal_detail(self.ir, self.result, self.timer_reads))
            detail = "I cannot schedule this as stated -- the timing is inconsistent: " + self.result.explanation()
            return _response(409, "INFEASIBLE_SCHEDULE", "TEMPORAL_CONTRADICTION", detail)

        steps = args.get("steps") if isinstance(args, dict) else None
        start_time = args.get("start_time") if isinstance(args, dict) else None
        if not isinstance(steps, list) or not isinstance(start_time, str):
            return self._commit(args, real_tool)  # malformed call: SimuHome's own validation answers it
        try:
            start_min = clock_to_minutes(self.base_time, start_time)
        except Exception:
            return self._commit(args, real_tool)

        devices = devices_in_steps(steps)
        candidates = [g for g in self.ir.goals if any(t.device_id in devices for t in g.targets)]
        if not candidates:
            self.stats.unmatched_calls += 1

        # Dry run first (C2): its result is also what tells us which goal this call is for (F3).
        dr = None
        dry_on = self.cfg.dry_run != "off" and self.dry_run_fn is not None
        if dry_on and start_min > self.cfg.dry_run_horizon_minutes:
            self.stats.events.append(("dry_run_skipped_beyond_horizon", start_time))
        elif dry_on:
            expected_all = [(t.device_id, a.attribute, a.value) for g in candidates for t in g.targets
                            if t.device_id in devices for a in t.asserts]
            dr = self._dry_run(start_time, steps, expected_all, check_effect=self.cfg.no_effect_check != "off")
            if dr is not None and dr.status == "failed":
                if self.cfg.dry_run == "reject" and self.stats.dry_run_rejections < self.cfg.max_plan_rejections:
                    self.stats.dry_run_rejections += 1
                    why = dr.error_detail or dr.error
                    return _response(409, "DRY_RUN_FAILED", "WORKFLOW_STEP_WOULD_FAIL", (
                        f"Scheduling refused: when this workflow runs at {start_time}, a step would fail: "
                        f"{why}. Fix the step (use exactly these command and argument names) and schedule again."
                    ))

        # F3: which goal is this call for? The one whose target values the steps actually produce.
        matched, satisfied = candidates, 0
        # Only a workflow that ran and changed something says what it is for; after a failed or no-op
        # dry run the "resulting" state is just the earlier state of the device.
        if dr is not None and candidates and dr.status == "completed" and dr.no_effect is not True:
            def score(g):
                # count only values this call CHANGES (differs from the twin copy without it) and that the goal
                # asks for: a pause sets "paused"; the dryness level the dryer already had says nothing
                n = 0
                for t in g.targets:
                    if t.device_id not in devices:
                        continue
                    for a in t.asserts:
                        key = (t.device_id, a.attribute)
                        now = dr.observed.get(key, object())
                        changed = (key not in dr.observed_without) or dr.observed_without[key] != now
                        n += bool(changed and values_match(a.value, now))
                return n
            scored = [(score(g), g) for g in candidates]
            satisfied = max(sc for sc, _ in scored)
            if satisfied > 0:
                matched = [g for sc, g in scored if sc == satisfied]
                self.stats.effect_matched_calls += 1

        # F4: the workflow changes nothing, and the devices are not already as the request wants.
        def fully_met(g):
            reqs = [(t.device_id, a) for t in g.targets if t.device_id in devices for a in t.asserts]
            return bool(reqs) and all(values_match(a.value, dr.observed.get((d, a.attribute), object())) for d, a in reqs)
        # "already as requested" = every value of some goal already holds (a redundant but harmless call)
        already = bool(dr is not None and dr.no_effect and any(fully_met(g) for g in candidates))
        if (dr is not None and dr.no_effect and candidates and not already and self.cfg.no_effect_check == "reject"
                and self.stats.no_effect_rejections < self.cfg.max_plan_rejections):
            self.stats.no_effect_rejections += 1
            states = "; ".join(f"{device_label(d)}: " + ", ".join(f"{k} = {v}" for k, v in st.items())
                               for d, st in dr.device_states.items()) or "unchanged"
            return _response(409, "NO_EFFECT", "WORKFLOW_WOULD_CHANGE_NOTHING", (
                f"Scheduling refused: at {start_time} these steps would change nothing ({states}). "
                "The device is probably not in the state you expect by then (for example a machine that never "
                "started cannot be paused). Check its current state, fix what is missing, and schedule again."
            ))

        # C1: the start time against the matched goal(s). SimuHome scores a goal as met if the requested
        # state holds at ANY moment of the goal's window (+/- tolerance); it does not care when the action
        # happened. So a call is refused only if it is too LATE (after the window ends) or so early that it
        # would overwrite an EARLIER goal on the same device before that goal's window opens.
        if matched and self.windows and satisfied == 0 and self.cfg.dry_run != "off":
            # G4: the dry run ran but the call produces no goal's values: the parser may have missed this goal,
            # so the other goals on the device say nothing about when THIS call should run. Not time-checked.
            self.stats.time_unchecked += 1
            matched = []
        if matched and self.windows:
            inside = [g for g in matched if self._time_allows(g, start_min)]
            if not inside:
                if self.stats.time_rejections < self.cfg.max_plan_rejections:
                    self.stats.time_rejections += 1
                    return _response(409, "TIME_MISMATCH", "START_TIME_DOES_NOT_MATCH_REQUEST",
                                     self._time_mismatch_detail(start_time, matched))
                self.stats.fail_open += 1
            else:
                matched = inside

        # target check (warn / reject) against the matched goal only
        notes = []
        if dr is not None and dr.status == "failed":
            notes.append(f"Dry run: a step will fail at {start_time}: {dr.error_detail or dr.error}.")
        if dr is not None and self.cfg.target_check != "off" and candidates:
            best = self._closest_goals(matched, start_min)
            expected = self._expected_targets(best, devices)
            miss = [(d, a, v, dr.observed.get((d, a))) for d, a, v in expected
                    if not values_match(v, dr.observed.get((d, a), object()))]
            if miss:
                msg = "; ".join(f"{device_label(d)} {a} would be {act!r}, the request asks for {v!r}"
                                for d, a, v, act in miss)
                if self.cfg.target_check == "reject" and self.stats.target_rejections < self.cfg.max_plan_rejections:
                    self.stats.target_rejections += 1
                    return _response(409, "TARGET_MISMATCH", "WORKFLOW_WOULD_NOT_REACH_REQUESTED_STATE",
                                     f"Scheduling refused: after this workflow runs at {start_time}, {msg}. "
                                     "Fix the steps and schedule again.")
                self.stats.target_warnings += 1
                notes.append(f"Dry run after {start_time}: {msg}.")
        resp = self._commit(args, real_tool)
        if notes and _ok(resp) and isinstance(resp.get("data"), dict):
            resp = copy.deepcopy(resp)
            resp["data"]["verifier_note"] = " ".join(notes)
        return resp

    def _earlier_bound(self, goal) -> tuple[Optional[float], bool]:
        """(earliest time a call for `goal` may run without undoing an EARLIER goal on one of its devices before
        that goal is checked, whether that bound is a deadline). SimuHome checks an ordinary goal at any moment
        of its window, so the earlier goal's state only has to exist once: its window START. A deadline goal
        ("still running 1 min before the dryer finishes", relation "by") is checked AT the deadline, so its
        state must last until then: its resolved time (G1, v2.2; v2.1 used the window start, which for a
        deadline is "now", so the rule never fired)."""
        resolved = {g.goal_id: g.resolved_minutes for g in self.result.goal_results}
        mine = resolved.get(goal.goal_id)
        devs = {t.device_id for t in goal.targets}
        best: tuple[Optional[float], bool] = (None, False)
        for g in self.ir.goals:
            if g.goal_id == goal.goal_id or not devs & {t.device_id for t in g.targets}:
                continue
            r = resolved.get(g.goal_id)
            w = self.windows.get(g.goal_id)
            if mine is None or r is None or r >= mine or w is None:
                continue
            b = (r, True) if w.deadline else ((w.lo, False) if w.lo is not None else (None, False))
            if b[0] is not None and (best[0] is None or b[0] > best[0]):
                best = b
        return best

    def _earlier_same_device_start(self, goal) -> Optional[float]:
        return self._earlier_bound(goal)[0]

    def _time_allows(self, goal, start_min: float) -> bool:
        w = self.windows.get(goal.goal_id)
        if w is None:
            return True
        slack = self.cfg.slack_minutes
        if w.hi is not None and start_min > w.hi + slack:
            return False  # too late: the state cannot hold within the goal's window
        lo, is_deadline = self._earlier_bound(goal)
        # a deadline is checked at one instant, so no slack before it (the agents' mistake is minutes early)
        if lo is not None and start_min < lo - (0.0 if is_deadline else slack):
            return False  # too early: it would replace the earlier goal's state before that goal is checked
        return True

    def evidence_steps(self) -> list[dict]:
        """F2 (answer mode): the tool calls the gate itself made for its verdict, as trajectory steps."""
        out = []
        for tool, args, resp in self.evidence_calls:
            read = self.timer_reads.get(args.get("device_id"))
            if read is not None and read.countdown_seconds is not None:
                thought = (f"Scheduling verifier: read {args.get('device_id')} OperationalState CountdownTime = "
                           f"{read.countdown_seconds:.0f} s, so it finishes at "
                           f"{minutes_to_clock(self.base_time, read.finish_minutes)}.")
            else:
                thought = f"Scheduling verifier: read {args.get('device_id')} OperationalState CountdownTime (no cycle running)."
            out.append({"thought": thought, "action": tool, "action_input": args, "observation": resp})
        return out

    def _closest_goals(self, goals, start_min: float):
        """Per device, keep only the goal whose computed time is closest to start_min.
        (Two goals on one device, e.g. dimmer to 30% at +13 min and to 20% at +21 min,
        must not both be expected from a single call.)"""
        resolved = {g.goal_id: g.resolved_minutes for g in self.result.goal_results}
        best: dict[str, Any] = {}
        for g in goals:
            dist = abs(resolved[g.goal_id] - start_min) if resolved.get(g.goal_id) is not None else float("inf")
            for t in g.targets:
                if t.device_id not in best or dist < best[t.device_id][0]:
                    best[t.device_id] = (dist, g)
        return best

    @staticmethod
    def _expected_targets(best: dict, devices: list[str]) -> list[tuple[str, str, Any]]:
        out = []
        for dev in devices:
            if dev in best:
                g = best[dev][1]
                out.extend((t.device_id, a.attribute, a.value) for t in g.targets if t.device_id == dev for a in t.asserts)
        return out

    def _dry_run(self, start_time: str, steps: list, expected: list, check_effect: bool = False):
        t0 = time.time()
        try:
            snapshot = _tool_data(self.call_tool("get_home_state", {}))
            if not isinstance(snapshot, dict):
                raise RuntimeError("no home snapshot")
            dr = self.dry_run_fn(snapshot, {"start_time": start_time, "steps": steps}, self._pending_committed(), expected,
                                 check_effect=check_effect)
            if dr.status == "error":
                self.stats.events.append(("dry_run_error", (dr.error or "")[:200]))
                self.stats.fail_open += 1
                return None
            return dr
        except Exception as e:
            self.stats.events.append(("dry_run_error", repr(e)[:200]))
            self.stats.fail_open += 1
            return None
        finally:
            self.stats.dry_run_seconds += time.time() - t0

    def _pending_committed(self) -> list[dict]:
        """Workflows this episode scheduled that are still pending in the real home. Ones the
        agent cancelled (cancel_workflow) or that already ran (their effect is in the snapshot)
        must not be replayed on the copy."""
        out = []
        for wf in self.committed:
            wid = wf.get("workflow_id")
            if wid is None:
                out.append(wf)
                continue
            status = (_tool_data(self.call_tool("get_workflow_status", {"workflow_id": wid})) or {}).get("status")
            if status in (None, "pending"):
                out.append(wf)
        return out

    def _time_mismatch_detail(self, start_time: str, goals) -> str:
        parts = []
        for g in goals:
            w = self.windows.get(g.goal_id)
            if w is None:
                continue
            earlier, earlier_deadline = self._earlier_bound(g)
            lo = minutes_to_clock(self.base_time, earlier) if earlier is not None else None
            hi = minutes_to_clock(self.base_time, w.hi) if w.hi is not None else None
            devs = ", ".join(device_label(t.device_id) for t in g.targets)
            why = [f"{device_label(r.anchor.device_id)} finishes at "
                   f"{minutes_to_clock(self.base_time, self.timer_reads[r.anchor.device_id].finish_minutes)}"
                   for r in g.readings if r.anchor.kind == "device_event"
                   and self.timer_reads.get(r.anchor.device_id) is not None
                   and self.timer_reads[r.anchor.device_id].finish_minutes is not None]
            if lo and hi and earlier_deadline:
                rng = (f"between {lo} and {hi} (not before {lo}: until then the device must stay as the earlier "
                       "part of the request says)")
            elif lo and hi:
                rng = f"no later than {hi} (and not before {lo}, when the previous step on this device is due)"
            elif hi:
                rng = f"no later than {hi}"
            else:
                rng = f"no earlier than {lo}"
            parts.append(f"for {devs} the request works out to {rng}" + (f" ({'; '.join(why)})" if why else ""))
        return (
            f"Scheduling refused: start_time {start_time} does not match the request ({'; '.join(parts)}). "
            "Recompute the time from the request and the current device states, then schedule again."
        )

    def _commit(self, args: dict, real_tool: Callable[[dict], dict]) -> dict:
        resp = real_tool(args)
        if _ok(resp):
            self.stats.passed += 1
            if isinstance(args, dict) and isinstance(args.get("steps"), list) and isinstance(args.get("start_time"), str):
                wid = (resp.get("data") or {}).get("workflow_id") if isinstance(resp.get("data"), dict) else None
                self.committed.append({"start_time": args["start_time"], "steps": args["steps"], "workflow_id": wid})
        return resp
