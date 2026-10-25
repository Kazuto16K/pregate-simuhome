"""Component 4: dry run of a proposed workflow on a copy of the home.

SimuHome's `schedule_workflow` only checks the tool name and that the start time
is in the future. Whether the steps can actually execute (right device, command,
cluster, argument names, writable attribute) is discovered only when the workflow
fires, long after the agent has called `finish`. This module finds out at
scheduling time:

    live snapshot (GET /home/state)
      -> SimulationConfig with base_time = the snapshot's current time
      -> a fresh, thread-less Home (SimuHome's own initializer)
      -> re-schedule the workflows already committed in this episode
      -> schedule the candidate, fast-forward to its due tick (+1)
      -> read the workflow status and the target attributes

Everything goes through SimuHome's own code (`_initialize_home_locally`,
`_convert_snapshot_to_simulation_config`, `Home._fast_forward_to`), so nothing in
SimuHome is edited (its licence is CC BY-NC-ND 4.0). This is a lightweight form of
the "simulation-based pre-validation" the SimuHome paper lists as future work.

Imports SimuHome lazily, so the rest of the package works without it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TargetMismatch:
    device_id: str
    attribute: str
    expected: Any
    actual: Any


@dataclass
class DryRunResult:
    ok: bool  # workflow completed in the dry run
    status: str  # completed | failed | pending | not_scheduled | error
    error: Optional[str] = None
    target_mismatches: list[TargetMismatch] = field(default_factory=list)
    seconds: float = 0.0
    # (device_id, "1.Cluster.Attribute") -> value right after the workflow ran (for every expected target)
    observed: dict = field(default_factory=dict)
    # True if the workflow changed nothing on the devices it touches (compared with an identical copy
    # of the house where it is not scheduled); None if not checked
    no_effect: Optional[bool] = None
    # the touched devices' state at that moment, for the refusal message
    device_states: dict = field(default_factory=dict)
    # same keys as `observed`, on the twin copy WITHOUT the workflow (only when check_effect)
    observed_without: dict = field(default_factory=dict)
    # v2.2: for a failed step, SimuHome's full reason (its workflow record keeps only the short message, e.g.
    # "Invalid arguments for command 'MoveToLevel'") plus the cluster's real commands and their arguments
    error_detail: Optional[str] = None


def _split_attribute(path: str) -> tuple[int, str, str]:
    endpoint, cluster, attribute = path.split(".", 2)
    return int(endpoint), cluster, attribute


def values_match(expected: Any, actual: Any) -> bool:
    """Plain equality, the same rule SimuHome's QT4 scorer uses (`_verify_asserts`).
    Values are raw attribute values (e.g. LevelControl.CurrentLevel 30), not percentages."""
    return actual == expected


def build_shadow_home(snapshot: dict):
    """A thread-less SimuHome Home whose tick 0 is the snapshot's current time."""
    from src.pipelines.episode_generation.core.home_augmenter import (  # SimuHome's own helpers
        _convert_snapshot_to_simulation_config,
        _initialize_home_locally,
    )

    config = _convert_snapshot_to_simulation_config(snapshot, base_time=snapshot["current_time"])
    home, _ = _initialize_home_locally(config)
    return home


_STATE_ATTRS = ("1.OnOff.OnOff", "1.OperationalState.OperationalState", "1.LevelControl.CurrentLevel",
                "1.FanControl.PercentSetting")


def _touched_devices(steps: list) -> list[str]:
    out = []
    for st in steps or []:
        dev = (st.get("args") or {}).get("device_id") if isinstance(st, dict) else None
        if isinstance(dev, str) and dev not in out:
            out.append(dev)
    return out


def _device_attrs(home, device_id: str):
    got = home._get_all_attributes(device_id)  # noqa: SLF001
    return got.data if got.success else None


def _readable_state(home, device_id: str) -> dict:
    out = {}
    for path in _STATE_ATTRS:
        try:
            endpoint, cluster, attribute = _split_attribute(path)
            got = home._get_attribute(device_id, endpoint, cluster, attribute)  # noqa: SLF001
            if got.success and got.data is not None:
                out[f"{cluster}.{attribute}"] = got.data
        except Exception:
            pass
    return out


def _signature(func) -> str:
    """'MoveToLevel(Level: int, TransitionTime: Optional[int] = None, ...)' from SimuHome's command handler."""
    import inspect

    try:
        params = [p for name, p in inspect.signature(func).parameters.items() if name != "self"]
    except (TypeError, ValueError):
        return "(...)"
    out = []
    for p in params:
        txt = p.name
        if p.annotation is not inspect.Parameter.empty:
            ann = p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, "__name__", str(p.annotation))
            txt += f": {str(ann).replace('typing.', '')}"
        if p.default is not inspect.Parameter.empty:
            txt += f" = {p.default!r}"
        out.append(txt)
    return "(" + ", ".join(out) + ")"


def cluster_help(home, device_id: str, endpoint_id: Any, cluster_id: str) -> Optional[str]:
    """The commands (with argument names, exactly as SimuHome's handlers take them) and the writable
    attributes of one cluster. `get_device_structure` lists command NAMES only, so an agent that guesses
    `{"level": 60}` for MoveToLevel (the handler takes `Level`) gets "Invalid arguments" and nothing else."""
    try:
        device = home.devices_by_id[device_id][1]
        cluster = device.endpoints[int(endpoint_id)][cluster_id]
    except Exception:
        return None
    cmds = [f"{name}{_signature(fn)}" for name, fn in cluster.commands.items()]
    writable = [a for a in cluster.attributes if a not in getattr(cluster, "readonly_attributes", set())]
    parts = []
    if cmds:
        parts.append(f"{cluster_id} commands: " + "; ".join(cmds))
    if writable:
        parts.append(f"writable {cluster_id} attributes: " + ", ".join(writable))
    return ". ".join(parts) if parts else None


_STEP_SHAPE = (
    'A step is {"tool": "execute_command", "args": {"device_id", "endpoint_id", "cluster_id", "command_id", '
    '"args": {...}}} or {"tool": "write_attribute", "args": {"device_id", "endpoint_id", "cluster_id", '
    '"attribute_id", "value"}}'
)


def explain_failure(home, steps: list, error_message: Optional[str]) -> Optional[str]:
    """Find the step that failed (re-running each step on the copy AFTER the failure, where the earlier
    steps succeed again and the bad one fails the same way) and return SimuHome's full reason for it, plus
    the cluster's commands/arguments when the step named a wrong command, wrong arguments or a read-only
    attribute."""
    for st in steps or []:
        if not isinstance(st, dict) or not isinstance(st.get("args"), dict):
            continue
        a = st["args"]
        try:
            if st.get("tool") == "execute_command":
                r = home._execute_command(**a)  # noqa: SLF001 (SimuHome's own command path)
            elif st.get("tool") == "write_attribute":
                r = home._write_attribute(**a)  # noqa: SLF001
            else:
                continue
        except TypeError as e:  # the step's own keys are wrong (e.g. "command" instead of "command_id")
            return f"the step's fields are wrong ({e}). {_STEP_SHAPE}"
        except Exception:
            continue
        if r.success or (error_message and r.error_message != error_message):
            continue
        detail = getattr(r, "error_detail", None)
        out = [f"{r.error_message}" + (f" ({detail})" if detail and detail != r.error_message else "")]
        msg = (r.error_message or "") + " " + (detail or "")
        if any(k in msg for k in ("Invalid arguments", "not found", "read-only", "readonly", "read only")):
            helptext = cluster_help(home, a.get("device_id"), a.get("endpoint_id"), a.get("cluster_id"))
            if helptext:
                out.append(helptext)
        return ". ".join(out)
    return None


def dry_run(
    snapshot: dict,
    candidate: dict,
    committed: Optional[list[dict]] = None,
    expected_targets: Optional[list[tuple[str, str, Any]]] = None,
    check_effect: bool = False,
) -> DryRunResult:
    """Run `candidate` = {"start_time", "steps"} on a copy of the home.

    `committed`: workflows the agent already scheduled in this episode (same
    shape); replayed first so the copy sees their effects. `expected_targets`:
    (device_id, "1.Cluster.Attribute", value) triples to read after the candidate
    ran; mismatches are reported, never raised.
    """
    t0 = time.time()
    try:
        home = build_shadow_home(snapshot)
    except Exception as e:  # snapshot the initializer cannot rebuild: say so, don't guess
        return DryRunResult(ok=False, status="error", error=f"shadow build failed: {e!r}"[:300], seconds=time.time() - t0)

    for wf in committed or []:
        home.schedule_workflow(wf["start_time"], wf["steps"])  # ones already in the past are skipped by Home

    res = home.schedule_workflow(candidate["start_time"], candidate["steps"])
    if not res.success:
        return DryRunResult(ok=False, status="not_scheduled", error=res.error_message, seconds=time.time() - t0)
    wf_id = res.data["workflow_id"]
    wf = home.workflows_by_id[wf_id]

    ff = home._fast_forward_to(int(wf["due_tick"]) + 1)  # noqa: SLF001 (SimuHome's own fast-forward)
    if not ff.success:
        return DryRunResult(ok=False, status="error", error=f"fast-forward failed: {ff.error_message}", seconds=time.time() - t0)

    mismatches, observed = [], {}
    for device_id, attr_path, expected in expected_targets or []:
        try:
            endpoint, cluster, attribute = _split_attribute(attr_path)
            got = home._get_attribute(device_id, endpoint, cluster, attribute)  # noqa: SLF001
            actual = got.data if got.success else None
        except Exception:
            actual = None
        observed[(device_id, attr_path)] = actual
        if not values_match(expected, actual):
            mismatches.append(TargetMismatch(device_id, attr_path, expected, actual))

    touched = _touched_devices(candidate.get("steps"))
    no_effect, states, observed_without = None, {}, {}
    if check_effect and touched:
        # Same copy of the house and the same moment, but WITHOUT the candidate workflow: if every touched
        # device ends up identical, the workflow does nothing (e.g. pausing a washer that is not running).
        try:
            twin = build_shadow_home(snapshot)
            for wf2 in committed or []:
                twin.schedule_workflow(wf2["start_time"], wf2["steps"])
            twin._fast_forward_to(int(wf["due_tick"]) + 1)  # noqa: SLF001
            no_effect = all(_device_attrs(home, d) == _device_attrs(twin, d) for d in touched)
            states = {d: _readable_state(home, d) for d in touched}
            for device_id, attr_path, _ in expected_targets or []:
                endpoint, cluster, attribute = _split_attribute(attr_path)
                got = twin._get_attribute(device_id, endpoint, cluster, attribute)  # noqa: SLF001
                observed_without[(device_id, attr_path)] = got.data if got.success else None
        except Exception:
            no_effect = None

    status = wf.get("status", "pending")
    error_detail = None
    if status == "failed":
        try:
            error_detail = explain_failure(home, candidate.get("steps"), wf.get("error"))
        except Exception:
            error_detail = None
    return DryRunResult(
        ok=status == "completed",
        status=status,
        error=wf.get("error"),
        target_mismatches=mismatches,
        seconds=time.time() - t0,
        observed=observed,
        no_effect=no_effect,
        device_states=states,
        observed_without=observed_without,
        error_detail=error_detail,
    )
