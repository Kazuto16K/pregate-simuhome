"""Component 2: IR -> Z3 constraints over real-valued time; SAT/UNSAT and time windows.

Time is one Real variable per goal: minutes since the episode's `base_time` (the
moment the user spoke, "now" = 0). Each TimeReading adds one constraint.

Changes from approach 1 (which solved every goal separately):
  - All goals are encoded in ONE solver, so readings can refer to each other:
    goal anchors ("8 min after the previous action") and a device anchor shared
    by several goals ("10 and 20 min after the dishwasher finishes" must agree on
    when the dishwasher finishes).
  - "by" readings are deadlines (t <= target + tol), not points.
  - When the request is consistent, `goal_windows` returns the interval [lo, hi]
    of start times each goal allows. The gate checks the agent's
    `schedule_workflow(start_time=...)` against these windows. This is the part a
    one-line Python comparison cannot do once goals chain off each other.

CPU-only; no GPU, simulator or SLM needed to run or test this module.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import z3

from .ir_schema import Goal, IntentIR, TimeReading

# Z3's default context is not thread-safe: two episodes checking at the same time
# (the parallel runner) crash it with "unreachable" / "invalid argument". Every
# public entry point holds this lock; a check takes milliseconds, so it costs nothing.
_Z3_LOCK = threading.RLock()


def _z3_serialized(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _Z3_LOCK:
            return fn(*args, **kwargs)
    return wrapper


def _parse_hms_to_minutes_of_day(hms: str) -> float:
    parts = [int(x) for x in hms.split(":")]
    h, m = parts[0], parts[1]
    s = parts[2] if len(parts) > 2 else 0
    return h * 60.0 + m + s / 60.0


def _is_minute_granular(hms: str) -> bool:
    return len(hms.split(":")) < 3


def _base_time_minutes_of_day(base_time: str) -> float:
    return _parse_hms_to_minutes_of_day(base_time.split(" ", 1)[1])


def minutes_to_clock(base_time: str, minutes: float) -> str:
    """Minutes since base_time -> "HH:MM:SS" wall-clock string."""
    base = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
    return (base + timedelta(minutes=float(minutes))).strftime("%H:%M:%S")


def clock_to_minutes(base_time: str, timestamp: str) -> float:
    """"YYYY-MM-DD HH:MM:SS" -> minutes since base_time."""
    base = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
    return (datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") - base).total_seconds() / 60.0


def reading_to_absolute_minutes(reading: TimeReading, base_time: str) -> Optional[float]:
    """Minutes since base_time for a self-contained clock-time reading, else None."""
    if reading.absolute_time is not None and reading.anchor.kind == "now":
        # Clock times are same-day in SimuHome QT4 episodes; no midnight wrap handling.
        return _parse_hms_to_minutes_of_day(reading.absolute_time) - _base_time_minutes_of_day(base_time)
    return None


@dataclass
class GoalCheckResult:
    goal_id: int
    sat: bool
    reason: Optional[str] = None  # populated on UNSAT (or "skipped: ...")
    unsat_core: Optional[list[str]] = None
    resolved_minutes: Optional[float] = None  # one consistent time, on SAT


@dataclass
class GoalWindow:
    """Start times (minutes since base_time) a goal allows. None = unbounded."""

    goal_id: int
    lo: Optional[float]
    hi: Optional[float]
    deadline: bool = False  # only an upper bound was stated ("by")

    def contains(self, minutes: float, slack: float = 0.0) -> bool:
        if self.lo is not None and minutes < self.lo - slack:
            return False
        if self.hi is not None and minutes > self.hi + slack:
            return False
        return True


@dataclass
class FeasibilityResult:
    feasible: bool
    goal_results: list[GoalCheckResult]
    base_time: Optional[str] = None
    core_descriptions: list[str] = field(default_factory=list)

    def explanation(self) -> str:
        bad = [g for g in self.goal_results if not g.sat]
        if not bad:
            return "All scheduling constraints are consistent."
        lines = [f"Goal {g.goal_id}: {g.reason}" for g in bad]
        return "Infeasible schedule - " + "; ".join(lines)


def _describe_reading(reading: TimeReading, base_time: Optional[str], value: Optional[float]) -> str:
    when = f" (= {minutes_to_clock(base_time, value)})" if (base_time and value is not None) else ""
    if reading.absolute_time is not None:
        return f"stated clock time {reading.absolute_time}"
    if reading.offset_minutes is None:
        return "unspecified time reading"
    if reading.anchor.kind == "now":
        anchor_desc = "now"
    elif reading.anchor.kind == "goal":
        anchor_desc = f"goal {reading.anchor.goal_ref}"
    else:
        anchor_desc = f"{reading.anchor.device_id} finishing"
    rel = "before" if reading.relation == "by" else reading.relation
    prefix = "no later than " if reading.relation == "by" else ""
    return f"{prefix}{reading.offset_minutes:g} min {rel} {anchor_desc}{when}"


class _Encoding:
    """Builds the joint constraint system once; reused for the check and the windows."""

    def __init__(self, ir: IntentIR):
        self.ir = ir
        self.t = {g.goal_id: z3.Real(f"t_goal_{g.goal_id}") for g in ir.goals}
        self.anchors: dict[str, z3.ArithRef] = {}
        self.pins: list[tuple[str, z3.BoolRef]] = []  # known anchor times (not "readings")
        self.readings: list[tuple[str, z3.BoolRef, int, TimeReading, Optional[float]]] = []
        self.skipped: dict[int, str] = {}
        for goal in ir.goals:
            self._encode_goal(goal)

    def _anchor_var(self, device_id: str, resolved: Optional[float], label: str) -> z3.ArithRef:
        if device_id not in self.anchors:
            self.anchors[device_id] = z3.Real(f"anchor_{device_id}")
        var = self.anchors[device_id]
        if resolved is not None:
            self.pins.append((f"{label}_anchor_known", var == z3.RealVal(resolved)))
        return var

    def _encode_goal(self, goal: Goal) -> None:
        if goal.requires_duration_model:
            self.skipped[goal.goal_id] = "skipped: requires a device duration model (not checked)"
            return
        t = self.t[goal.goal_id]
        for i, reading in enumerate(goal.readings):
            label = f"g{goal.goal_id}_r{i}"
            extra = 1.0 if (reading.absolute_time and _is_minute_granular(reading.absolute_time)) else 0.0
            # An "HH:MM" clock time is only known to the minute, so widen by one minute.
            tol = z3.RealVal(reading.tolerance_minutes + extra)
            sign = -1 if reading.relation in ("before", "by") else 1
            nominal: Optional[float] = None  # the reading's own time, when known, for explanations

            if reading.anchor.kind == "device_event":
                anchor_var = self._anchor_var(reading.anchor.device_id or "?", reading.resolved_at_minutes, label)
                if reading.offset_minutes is None:
                    continue
                offset = 0.0 if reading.relation == "at" else sign * reading.offset_minutes
                target = anchor_var + z3.RealVal(offset)
                if reading.resolved_at_minutes is not None:
                    nominal = reading.resolved_at_minutes + offset
            elif reading.anchor.kind == "goal":
                ref = reading.anchor.goal_ref
                if ref is None or ref not in self.t or ref == goal.goal_id or reading.offset_minutes is None:
                    continue  # malformed reference (e.g. a parser slip): no constraint rather than a wrong one
                target = self.t[ref] + z3.RealVal(sign * reading.offset_minutes)
            else:
                resolved = reading_to_absolute_minutes(reading, self.ir.base_time)
                if resolved is not None:
                    target = z3.RealVal(resolved)
                    nominal = resolved
                elif reading.offset_minutes is not None:
                    target = z3.RealVal(sign * reading.offset_minutes)
                    nominal = sign * reading.offset_minutes
                else:
                    continue

            if reading.relation == "by":
                constraint = t <= target + tol
            else:
                constraint = z3.And(t >= target - tol, t <= target + tol)
            self.readings.append((label, constraint, goal.goal_id, reading, nominal))

    def solver(self, track: bool) -> z3.Solver:
        s = z3.Solver()
        for label, c in self.pins:
            if track:
                s.assert_and_track(c, z3.Bool(label))
            else:
                s.add(c)
        for label, c, *_ in self.readings:
            if track:
                s.assert_and_track(c, z3.Bool(label))
            else:
                s.add(c)
        return s


@_z3_serialized
def check_feasibility(ir: IntentIR) -> FeasibilityResult:
    enc = _Encoding(ir)
    solver = enc.solver(track=True)
    if solver.check() == z3.sat:
        model = solver.model()
        results = []
        for g in ir.goals:
            if g.goal_id in enc.skipped:
                results.append(GoalCheckResult(goal_id=g.goal_id, sat=True, reason=enc.skipped[g.goal_id]))
                continue
            val = model.eval(enc.t[g.goal_id], model_completion=True)
            results.append(GoalCheckResult(goal_id=g.goal_id, sat=True, resolved_minutes=float(val.as_fraction())))
        return FeasibilityResult(feasible=True, goal_results=results, base_time=ir.base_time)

    core = {str(c) for c in solver.unsat_core()}
    by_label = {label: (gid, r, nominal) for label, _, gid, r, nominal in enc.readings}
    bad_goals: dict[int, list[str]] = {}
    descriptions = []
    # "*_anchor_known" pins only fix a symbolic anchor to a runtime-known value;
    # they are not a statement in the request, so they are not named.
    for label in sorted(core):
        if label in by_label:
            gid, r, nominal = by_label[label]
            d = _describe_reading(r, ir.base_time, nominal)
            bad_goals.setdefault(gid, []).append(d)
            descriptions.append(f"goal {gid}: {d}")
    results = []
    for g in ir.goals:
        if g.goal_id in bad_goals:
            results.append(
                GoalCheckResult(
                    goal_id=g.goal_id,
                    sat=False,
                    reason=" vs. ".join(bad_goals[g.goal_id]),
                    unsat_core=sorted(core),
                )
            )
        else:
            results.append(GoalCheckResult(goal_id=g.goal_id, sat=True, reason=enc.skipped.get(g.goal_id)))
    if not bad_goals:  # core made only of pins (should not happen); still report UNSAT honestly
        results[0] = GoalCheckResult(goal_id=results[0].goal_id, sat=False, reason="unsatisfiable", unsat_core=sorted(core))
    return FeasibilityResult(feasible=False, goal_results=results, base_time=ir.base_time, core_descriptions=descriptions)


def _bound(enc: _Encoding, var: z3.ArithRef, maximize: bool) -> Optional[float]:
    opt = z3.Optimize()
    for _, c in enc.pins:
        opt.add(c)
    for _, c, *_ in enc.readings:
        opt.add(c)
    h = opt.maximize(var) if maximize else opt.minimize(var)
    if opt.check() != z3.sat:
        return None
    value = opt.upper(h) if maximize else opt.lower(h)
    # Z3 hands back an integer or rational numeral when the bound is finite, and a
    # non-numeral term (oo, epsilon) when the goal is unbounded in that direction.
    if z3.is_int_value(value):
        return float(value.as_long())
    if z3.is_rational_value(value):
        return float(value.as_fraction())
    return None


@_z3_serialized
def goal_windows(ir: IntentIR) -> dict[int, GoalWindow]:
    """For a consistent IR: the [lo, hi] start-time interval of every goal.

    Goals whose readings are all unresolved (e.g. the anchor device is idle, so its
    finish time is unknown) come back unbounded, and the gate does not check them.
    """
    enc = _Encoding(ir)
    if enc.solver(track=False).check() != z3.sat:
        return {}
    windows = {}
    for g in ir.goals:
        if g.goal_id in enc.skipped:
            windows[g.goal_id] = GoalWindow(goal_id=g.goal_id, lo=None, hi=None)
            continue
        t = enc.t[g.goal_id]
        lo, hi = _bound(enc, t, maximize=False), _bound(enc, t, maximize=True)
        deadline = all(r.relation == "by" for r in g.readings) and bool(g.readings)
        windows[g.goal_id] = GoalWindow(goal_id=g.goal_id, lo=lo, hi=hi, deadline=deadline)
    return windows
