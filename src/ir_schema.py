"""Typed intermediate representation (IR v3) for SimuHome QT4 scheduling requests.

An IR describes one `schedule_workflow`-shaped user request as a set of goals.
Each goal is a set of target device assertions gated by one or more *time
readings* (ways the natural-language query says when the action happens).
A feasible request has one consistent reading per goal; an infeasible one states
two readings that resolve to different times for the same action (SimuHome's
"Contradiction Blindness" failure mode).

Changes from v2 (approach 1):
  - Anchor kind "goal": a reading measured from an EARLIER goal of the same
    request ("8 minutes after the previous action"). v2 had no such anchor and
    labelled these readings as "now"-anchored, which is harmless for a pure
    contradiction check but wrong for checking the agent's start_time.
  - Relation "by": a deadline (the state must hold no later than t). SimuHome
    marks these goals with tolerance_ticks == 0 and relation "before" (e.g. QT4-3
    "start the dryer now ... pause it when the washer finishes": the dryer only
    has to be running before the washer ends). v2 treated them as a point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Relation = Literal["after", "before", "at", "by"]
AnchorKind = Literal["now", "device_event", "goal"]

IR_VERSION = 3


@dataclass
class Anchor:
    """What a time offset is measured from."""

    kind: AnchorKind = "now"
    room_id: Optional[str] = None
    device_id: Optional[str] = None
    device_type: Optional[str] = None
    goal_ref: Optional[int] = None  # kind == "goal": index of the earlier goal

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "room_id": self.room_id,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "goal_ref": self.goal_ref,
        }


@dataclass
class TimeReading:
    """One way the query expresses *when* an action should occur.

    All times are minutes since the episode's `base_time` (the moment the user
    spoke). A second reading on the same goal that disagrees with the first is
    what makes a request infeasible.
    """

    relation: Relation
    anchor: Anchor
    offset_minutes: Optional[float] = None  # relative reading
    absolute_time: Optional[str] = None  # "HH:MM[:SS]", clock-time reading
    # device_event anchors only: when the anchor device finishes (minutes since
    # base_time). Never in the training target; filled from live simulator state.
    resolved_at_minutes: Optional[float] = None
    tolerance_minutes: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "anchor": self.anchor.to_dict(),
            "offset_minutes": self.offset_minutes,
            "absolute_time": self.absolute_time,
            "resolved_at_minutes": self.resolved_at_minutes,
            "tolerance_minutes": self.tolerance_minutes,
        }


@dataclass
class Assertion:
    attribute: str
    value: Any
    value_type: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "value_type": self.value_type,
            "description": self.description,
        }


@dataclass
class Target:
    room_id: str
    device_id: str
    device_type: Optional[str]
    asserts: list[Assertion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "asserts": [a.to_dict() for a in self.asserts],
        }


@dataclass
class Goal:
    """One scheduled effect: a set of device assertions gated by time reading(s)."""

    goal_id: int
    readings: list[TimeReading]
    targets: list[Target]
    # Set when the conflict is a duration/resource issue rather than a stated-time
    # mismatch (e.g. "finish the washer exactly when the dishwasher finishes" but the
    # washer cycle is longer). Not checkable without a device duration model.
    requires_duration_model: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "readings": [r.to_dict() for r in self.readings],
            "targets": [t.to_dict() for t in self.targets],
            "requires_duration_model": self.requires_duration_model,
        }


@dataclass
class IntentIR:
    query: str
    query_type: str
    case: Literal["feasible", "infeasible", "unknown"]
    base_time: str  # "YYYY-MM-DD HH:MM:SS"
    goals: list[Goal] = field(default_factory=list)
    conflict_type: Optional[str] = None  # ground-truth label, training-time only

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "case": self.case,
            "base_time": self.base_time,
            "goals": [g.to_dict() for g in self.goals],
            "conflict_type": self.conflict_type,
        }


def ir_from_dict(d: dict) -> IntentIR:
    """Inverse of IntentIR.to_dict() (used for the debug ground-truth IRs)."""
    goals = []
    for g in d["goals"]:
        readings = [
            TimeReading(anchor=Anchor(**r["anchor"]), **{k: v for k, v in r.items() if k != "anchor"})
            for r in g["readings"]
        ]
        targets = [
            Target(
                room_id=t["room_id"],
                device_id=t["device_id"],
                device_type=t.get("device_type"),
                asserts=[Assertion(**a) for a in t.get("asserts", [])],
            )
            for t in g["targets"]
        ]
        goals.append(
            Goal(
                goal_id=g["goal_id"],
                readings=readings,
                targets=targets,
                requires_duration_model=g.get("requires_duration_model", False),
            )
        )
    return IntentIR(
        query=d["query"],
        query_type=d["query_type"],
        case=d["case"],
        base_time=d["base_time"],
        goals=goals,
        conflict_type=d.get("conflict_type"),
    )


# ---------------------------------------------------------------------------
# Compact wire format (what the SLM is trained to emit)
# ---------------------------------------------------------------------------
# Keeps only what the query text determines: per goal, the time readings and the
# targets. Z3 decides feasibility; the model never says it. v3 adds `goal_ref`
# for goal anchors and the "by" relation; everything else is unchanged from v2,
# so a v2 parser's output still loads (it just never produces goal anchors).

DEFAULT_TOLERANCE_MINUTES = 0.5


def compact_ir_dict(ir_dict: dict, *, keep_resolved: bool = False) -> dict:
    """Full IR dict -> compact dict. Clock times are cut to HH:MM (the query
    states minutes, never the simulator's seconds)."""
    goals = []
    for g in ir_dict["goals"]:
        readings = []
        for r in g["readings"]:
            anchor = {"kind": r["anchor"]["kind"]}
            if r["anchor"].get("device_id"):
                anchor["device_id"] = r["anchor"]["device_id"]
            if r["anchor"].get("room_id"):
                anchor["room_id"] = r["anchor"]["room_id"]
            if r["anchor"].get("goal_ref") is not None:
                anchor["goal_ref"] = r["anchor"]["goal_ref"]
            out = {"relation": r["relation"], "anchor": anchor}
            if r.get("offset_minutes") is not None:
                out["offset_minutes"] = r["offset_minutes"]
            if r.get("absolute_time") is not None:
                out["absolute_time"] = ":".join(r["absolute_time"].split(":")[:2])
            if keep_resolved and r.get("resolved_at_minutes") is not None:
                out["resolved_at_minutes"] = r["resolved_at_minutes"]
            readings.append(out)
        targets = [
            {
                "device_id": t["device_id"],
                "asserts": [{"attribute": a["attribute"], "value": a["value"]} for a in t.get("asserts", [])],
            }
            for t in g["targets"]
        ]
        goals.append({"readings": readings, "targets": targets})
    return {"goals": goals}


def ir_from_compact(compact: dict, query: str, base_time: str) -> IntentIR:
    """Compact dict (+ the live query and the episode base_time) -> IntentIR."""
    goals = []
    for i, g in enumerate(compact["goals"]):
        readings = []
        for r in g["readings"]:
            a = r.get("anchor") or {}
            readings.append(
                TimeReading(
                    relation=r.get("relation", "after"),
                    anchor=Anchor(
                        kind=a.get("kind", "now"),
                        room_id=a.get("room_id"),
                        device_id=a.get("device_id"),
                        goal_ref=a.get("goal_ref"),
                    ),
                    offset_minutes=r.get("offset_minutes"),
                    absolute_time=r.get("absolute_time"),
                    resolved_at_minutes=r.get("resolved_at_minutes"),
                    tolerance_minutes=r.get("tolerance_minutes", DEFAULT_TOLERANCE_MINUTES),
                )
            )
        targets = [
            Target(
                room_id=t.get("room_id", ""),
                device_id=t["device_id"],
                device_type=None,
                asserts=[Assertion(attribute=a["attribute"], value=a["value"]) for a in t.get("asserts", [])],
            )
            for t in g.get("targets", [])
        ]
        goals.append(Goal(goal_id=i, readings=readings, targets=targets))
    return IntentIR(query=query, query_type="qt4", case="unknown", base_time=base_time, goals=goals)


def uses_goal_anchors(compact: dict) -> bool:
    """True if a compact IR contains any v3 goal anchor (i.e. came from a v3 parser
    or from v3 labels). Used to decide whether start_time checks are trustworthy."""
    return any(
        (r.get("anchor") or {}).get("kind") == "goal"
        for g in compact.get("goals", [])
        for r in g.get("readings", [])
    )
