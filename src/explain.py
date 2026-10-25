"""Plain-language explanations of why a request is impossible (fixes F1 and F2).

Session-1 traces showed two problems with the old one-line explanation
("24 min before utility_room_laundry_washer_1 finishing (= 18:26:21) vs. stated clock
time 18:13"): the agent read 18:26 as the washer's finish time and told the user so
(false; it finishes at 18:50), and QT4-3 replies said things like "no later than 1 min
before goal 1", which nobody can follow. SimuHome's judges for QT4-2/QT4-3 also require
the actual finish time read from the device's CountdownTime, and they see only the
agent's thoughts and final answer.

So every explanation here:
  - names devices in words ("utility room laundry washer 1"), never "goal 1";
  - states each device's real finish time and the timer reading it came from;
  - spells out the time each statement in the request works out to.

Pure functions over the IR, the feasibility result and the timer readings the gate took;
no SimuHome import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .constraint_encoder import FeasibilityResult, minutes_to_clock, reading_to_absolute_minutes
from .ir_schema import Goal, IntentIR, TimeReading


@dataclass
class TimerRead:
    """One CountdownTime reading the gate took from the simulator."""

    device_id: str
    countdown_seconds: Optional[float]  # None: the device reported no running cycle
    finish_minutes: Optional[float]  # minutes since base_time; None if idle


def device_label(device_id: str) -> str:
    """'utility_room_laundry_washer_1' -> 'utility room laundry washer 1'."""
    return device_id.replace("_", " ") if device_id else "the device"


def _clock(base_time: str, minutes: float) -> str:
    return minutes_to_clock(base_time, minutes)


def _hhmm(clock: str) -> str:
    return ":".join(clock.split(":")[:2])


_STATE_VERBS = {0: "stop", 1: "start", 2: "pause", 3: "resume"}


def _verb(attribute: str, value) -> Optional[str]:
    """What setting `attribute` to `value` means in words, for the common Matter attributes."""
    cluster_attr = ".".join(attribute.split(".")[1:])
    if cluster_attr == "OnOff.OnOff":
        return "turn on" if value else "turn off"
    if cluster_attr == "OperationalState.OperationalState" and value in _STATE_VERBS:
        return _STATE_VERBS[value]
    if cluster_attr == "LevelControl.CurrentLevel":
        return f"set to level {value}"
    if cluster_attr == "FanControl.PercentSetting":
        return f"set the fan to {value}%"
    return None


def _action(goal: Goal) -> str:
    """What a goal does, in words: 'start utility room laundry dryer 1', 'turn on kitchen light 1 and ...'."""
    parts = []
    for t in goal.targets:
        verbs = [v for v in (_verb(a.attribute, a.value) for a in t.asserts) if v]
        # an operational verb (start/pause/stop) says more than a power switch
        verb = next((v for v in verbs if v in _STATE_VERBS.values()), verbs[0] if verbs else "change")
        parts.append(f"{verb} {device_label(t.device_id)}")
    return " and ".join(parts) if parts else "the requested action"


def timer_evidence(reads: dict[str, TimerRead], base_time: str) -> list[str]:
    """One sentence per timer reading: what was read and what it means."""
    out = []
    for r in reads.values():
        if r.countdown_seconds is None or r.finish_minutes is None:
            out.append(f"{device_label(r.device_id)} is not running a cycle (its OperationalState CountdownTime "
                       "shows no remaining time), so its finish time is unknown.")
        else:
            mins = r.countdown_seconds / 60.0
            out.append(f"{device_label(r.device_id)} has {r.countdown_seconds:.0f} s (about {mins:.0f} min) left on its "
                       f"OperationalState CountdownTime, so it finishes at {_clock(base_time, r.finish_minutes)}.")
    return out


class _Resolver:
    """Works out the clock time each reading / goal refers to, where it is known."""

    def __init__(self, ir: IntentIR, reads: dict[str, TimerRead]):
        self.ir, self.reads = ir, reads
        self.base = ir.base_time
        self._goal_cache: dict[int, Optional[tuple[float, str]]] = {}

    def reading(self, r: TimeReading, depth: int = 0) -> tuple[Optional[float], str]:
        """(minutes or None, phrase) for one reading, e.g. (52.0, '8 min after X finishes')."""
        off = r.offset_minutes or 0.0
        sign = -1 if r.relation in ("before", "by") else 1
        when = "before" if sign < 0 else "after"
        a = r.anchor
        if r.absolute_time is not None and a.kind == "now":
            return reading_to_absolute_minutes(r, self.base), f"at {r.absolute_time} (the time you stated)"
        if a.kind == "now":
            return sign * off, f"{off:g} minutes from now"
        if a.kind == "device_event":
            dev = device_label(a.device_id or "")
            fin = r.resolved_at_minutes
            if fin is None and a.device_id in self.reads:
                fin = self.reads[a.device_id].finish_minutes
            phrase = f"when {dev} finishes" if off == 0 else f"{off:g} minutes {when} {dev} finishes"
            if fin is None:
                return None, phrase
            return fin + sign * off, phrase
        if a.kind == "goal" and a.goal_ref is not None and depth < 4:
            ref = self.goal(a.goal_ref, depth + 1)
            ref_goal = self.ir.goals[a.goal_ref] if a.goal_ref < len(self.ir.goals) else None
            ref_name = (_action(ref_goal) if ref_goal else "the other action").replace("start ", "starting ", 1)                 .replace("pause ", "pausing ", 1).replace("stop ", "stopping ", 1).replace("turn ", "turning ", 1)                 .replace("set ", "setting ", 1)
            if r.relation == "by":
                phrase = f"before {ref_name}"
            elif off == 0:
                phrase = f"together with {ref_name}"
            else:
                phrase = f"{off:g} minutes {when} {ref_name}"
            if ref is None:
                return None, phrase
            return ref[0] + sign * off, phrase + f" ({ref[1]}, at {_clock(self.base, ref[0])})"
        return None, "at an unspecified time"

    def goal(self, gid: int, depth: int = 0) -> Optional[tuple[float, str]]:
        """The time a goal happens according to its first reading that can be resolved."""
        if gid in self._goal_cache:
            return self._goal_cache[gid]
        self._goal_cache[gid] = None  # guards against cycles
        if not (0 <= gid < len(self.ir.goals)):
            return None
        for r in self.ir.goals[gid].readings:
            if r.relation == "by":
                continue  # a deadline does not fix a time
            m, phrase = self.reading(r, depth)
            if m is not None:
                self._goal_cache[gid] = (m, phrase)
                return self._goal_cache[gid]
        return None


def conflict_sentences(ir: IntentIR, result: FeasibilityResult, reads: dict[str, TimerRead]) -> list[str]:
    """One plain sentence per contradictory goal."""
    res = _Resolver(ir, reads)
    base = ir.base_time
    out, single = [], []
    for gr in result.goal_results:
        if gr.sat or not (0 <= gr.goal_id < len(ir.goals)):
            continue
        goal = ir.goals[gr.goal_id]
        parts, deadline = [], None
        for r in goal.readings:
            m, phrase = res.reading(r)
            if r.relation == "by" and m is not None:
                deadline = (m, phrase)
                continue
            parts.append(phrase if m is None or "(the time you stated)" in phrase
                         else f"{phrase}, which is {_clock(base, m)}")
        name = _action(goal)
        if deadline is not None:
            stated = " and ".join(parts) if parts else "at a later time"
            out.append(
                f"You asked to {name} {stated}, but that has to happen {deadline[1]}, that is no later than about "
                f"{_hhmm(_clock(base, deadline[0]))}. That moment comes earlier, so the request cannot be carried out "
                "as stated."
            )
        elif len(parts) >= 2:
            out.append(f"You asked to {name} {parts[0]}, and also {' and '.join(parts[1:])}. "
                       "These are different times, so both cannot be true.")
        elif parts:
            single.append(f"To {name} {parts[0]} does not fit the rest of the request.")
    if not out:
        out = single  # only when no clearer sentence exists
    if not out:
        out.append("Two parts of the request give the same action different times, so both cannot be true.")
    return out


def user_reply(ir: IntentIR, result: FeasibilityResult, reads: dict[str, TimerRead]) -> str:
    """Answer-mode reply to the user (the agent is not called)."""
    evidence = timer_evidence(reads, ir.base_time)
    text = "I can't schedule this as requested, because the timing contradicts itself. "
    if evidence:
        text += "I checked the devices: " + " ".join(evidence) + " "
    text += " ".join(conflict_sentences(ir, result, reads))
    text += " I have not scheduled or changed anything. Please tell me which time you want."
    return text


def agent_hint(ir: IntentIR, result: FeasibilityResult, reads: dict[str, TimerRead]) -> str:
    """Hint-mode note appended to the agent's first message."""
    lines = " ".join(conflict_sentences(ir, result, reads))
    evidence = " ".join(timer_evidence(reads, ir.base_time))
    check = ""
    timed = [r for r in reads.values() if r.countdown_seconds is not None]
    if timed:
        devs = "; ".join(
            f"get_attribute(device_id='{r.device_id}', endpoint_id=1, cluster_id='OperationalState', "
            "attribute_id='CountdownTime')" for r in timed)
        check = (f" Before answering, confirm the finish time yourself with {devs}, and state in your answer the "
                 "device's actual finish time you read.")
    return (
        "\n\n[Note from the scheduling verifier, computed before you started: this request contradicts itself. "
        f"{evidence + ' ' if evidence else ''}{lines}{check} Do not call schedule_workflow; call finish and explain "
        "the conflict to the user in plain words, with the times.]"
    )


def refusal_detail(ir: IntentIR, result: FeasibilityResult, reads: dict[str, TimerRead]) -> str:
    """409 detail when the agent tries to schedule a contradictory request anyway."""
    evidence = " ".join(timer_evidence(reads, ir.base_time))
    return (
        "The request itself is contradictory, so no start_time can satisfy it and changing the time will not help. "
        f"{evidence + ' ' if evidence else ''}{' '.join(conflict_sentences(ir, result, reads))} "
        "Do not call schedule_workflow again for this request; call finish and explain the conflict to the user."
    )
