"""Wires the approach-2 gate into SimuHome's ReAct agent, one gate per episode.

Two monkeypatches (SimuHome's source is CC BY-NC-ND 4.0, so it is never edited):
  - `ReActAgent.run`: builds the episode's EpisodeGate (parse once, pre-flight once)
    and, depending on the arm, appends a hint to the query, answers the user
    without calling the agent, or appends the prompt-only self-check instruction.
  - `TOOL_REGISTRY["schedule_workflow"]`: routes each call to the gate of the
    episode running in the CURRENT THREAD (several episodes run in parallel).

The gate's own tool calls (current time, CountdownTime, home snapshot) go through
SimuHome's `run_tool` in the same thread, so they reach that episode's simulator,
and they are not recorded as agent actions (scoring is unaffected).

Imports SimuHome lazily (call `install()` after SimuHome is on sys.path).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from .gate import ARMS, EpisodeGate, GateConfig

SELF_CHECK_PROMPT = (
    " IMPORTANT: before scheduling anything, work out the exact time of every requested action. If two parts of the"
    " request give the same event two different times (a contradiction), do NOT call schedule_workflow; call finish"
    " and explain the conflict."
)


class Harness:
    def __init__(
        self,
        parse_fn: Callable[[str], dict],
        dry_run_fn: Optional[Callable[..., Any]] = None,
        ir_version: int = 2,
        arms: Optional[dict[str, Optional[GateConfig]]] = None,
    ):
        self.parse_fn = parse_fn
        self.dry_run_fn = dry_run_fn
        self.ir_version = ir_version
        self.arms = dict(arms or ARMS)
        self.arm = "baseline"
        self._tls = threading.local()
        self._lock = threading.Lock()
        self.telemetry: list[dict] = []
        self._installed = False

    # ------------------------------------------------------------ public API
    def set_arm(self, arm: str) -> None:
        if arm not in self.arms:
            raise KeyError(f"unknown arm {arm!r}; known: {sorted(self.arms)}")
        self.arm = arm

    def begin_episode(self, key: str) -> None:
        """Called by the runner in the worker thread before each episode (labels telemetry)."""
        self._tls.episode = key

    def install(self) -> None:
        if self._installed:
            return
        from src.agents.strategies.base import AgentResult, AgentStep
        from src.agents.strategies.react_agent import ReActAgent
        from src.agents.tools import TOOL_REGISTRY, run_tool, tool_schedule_workflow

        self._AgentResult, self._AgentStep, self._run_tool = AgentResult, AgentStep, run_tool
        original_tool = tool_schedule_workflow
        original_run = getattr(ReActAgent.run, "_approach2_orig", ReActAgent.run)  # idempotent if re-installed
        harness = self

        def gated_schedule_workflow(args):
            gate = getattr(harness._tls, "gate", None)
            if gate is None:
                return original_tool(args)
            return gate.on_schedule(args, original_tool)

        # The agent's prompt lists each tool from its docstring; without this the agent
        # never sees schedule_workflow's argument description.
        gated_schedule_workflow.__doc__ = original_tool.__doc__
        TOOL_REGISTRY["schedule_workflow"] = gated_schedule_workflow

        def patched_run(agent_self, user_query, **kwargs):
            return harness._run_episode(original_run, agent_self, user_query, **kwargs)

        patched_run._approach2_orig = original_run
        ReActAgent.run = patched_run
        self._installed = True

    # --------------------------------------------------------------- internals
    def _parse(self, query: str) -> tuple[Optional[dict], float, Optional[str]]:
        t0 = time.time()
        try:
            return self.parse_fn(query), time.time() - t0, None
        except Exception as e:  # parse failure -> the gate fails open
            return None, time.time() - t0, repr(e)[:200]

    def _run_episode(self, original_run, agent_self, user_query: str, **kwargs):
        arm = self.arm
        cfg = self.arms[arm]
        query = user_query
        gate = None
        record = {"episode": getattr(self._tls, "episode", None), "arm": arm, "query": user_query[:200],
                  "answered_by_gate": False, "parse_seconds": 0.0, "parse_error": None}
        t0 = time.time()
        try:
            if cfg is not None:
                cfg = GateConfig(**{**cfg.__dict__, "ir_version": self.ir_version})
                compact, record["parse_seconds"], record["parse_error"] = self._parse(user_query)
                base_time = kwargs.get("current_time")
                if not base_time or base_time == "None":
                    base_time = (self._run_tool("get_current_time", {}).get("data") or {}).get("now")
                gate = EpisodeGate(user_query, base_time, compact, cfg, self._run_tool, self.dry_run_fn)
                self._tls.gate = gate
                action, text = gate.preflight_action()
                if action == "answer":
                    record["answered_by_gate"] = True
                    # F2: the readings the verdict rests on (real tool calls the gate made) come first,
                    # then the reply. They are not added to tool_calls: the agent did not make them.
                    steps = [self._AgentStep(step=i + 1, thought=e["thought"], action=e["action"],
                                             action_input=e["action_input"], observation=e["observation"])
                             for i, e in enumerate(gate.evidence_steps())]
                    steps.append(self._AgentStep(
                        step=len(steps) + 1,
                        thought="Scheduling verifier: the stated times contradict each other, so nothing is scheduled; "
                                "replying to the user with the times.",
                        action="finish", action_input={"answer": text}, observation=None))
                    return self._AgentResult(steps=steps, tool_calls=[], final_answer=text, raw_responses=[], events=[])
                if action == "hint":
                    query = query + text
            if arm == "selfcheck":
                query = query + SELF_CHECK_PROMPT
            return original_run(agent_self, query, **kwargs)
        finally:
            self._tls.gate = None
            record["episode_seconds"] = round(time.time() - t0, 2)
            if gate is not None:
                record.update(gate.stats.as_dict())
            with self._lock:
                self.telemetry.append(record)
