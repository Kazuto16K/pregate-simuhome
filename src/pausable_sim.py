"""SimuHome's simulator server with a clock that can be held while the agent waits for its LLM.

Why: SimuHome's home runs in REAL time (one 0.1 s tick per 0.1 s of wall clock; `Home.__simulation_loop`
sleeps between ticks) and the agent is told the current time once, in its first prompt. With a slow hosted
LLM (E1 session 2: 65-80 s per call, 6-13 min per episode) the agent's idea of "now" is minutes stale by the
time it schedules: start times are already in the past (HTTP 400) and "pause it when the dryer finishes"
lands 1-5 min early. That measures the API's latency, not the agent or the gate, and it penalises every extra
turn (a correct gate refusal costs one more LLM call). Holding the clock during LLM calls makes the result the
same as with an instant LLM, for every arm alike. Tool calls, the gate's own work and the scorer run with the
clock running as usual. The paper reports this as a latency-controlled evaluation.

How: run this file instead of `python -m src.simulator.api.app` (same port variable, cwd = SimuHome). It
imports SimuHome's app unchanged (CC BY-NC-ND 4.0: no file is edited), wraps the first per-tick method of
`Home` so a tick waits while the clock is held, and adds one route:

    POST /api/__clock__   {"hold": true | false}  ->  {"held": bool, "tick": int}

While held, the simulation thread is blocked, so the API must not be called (SimuHome answers API calls from
that thread): the runner holds the clock only around LLM requests, when the agent's thread is waiting anyway.
A hold longer than MAX_HOLD_SECONDS is released automatically, so a crashed client cannot freeze a server.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

MAX_HOLD_SECONDS = float(os.environ.get("SIM_MAX_HOLD_SECONDS", "1800"))

_running = threading.Event()
_running.set()
_held_since = [0.0]
_stats = {"holds": 0, "auto_released": 0, "held_seconds": 0.0}


def _install() -> object:
    sys.path.insert(0, os.getcwd())  # cwd = SimuHome checkout; its package is `src`
    from src.simulator.domain import home as home_mod

    Home = home_mod.Home
    name = "_Home__process_time_aware_devices"  # the first step of every tick in Home.__simulation_loop
    orig = getattr(Home, name)

    def gated(self, *a, **kw):
        if not _running.is_set():
            if not _running.wait(MAX_HOLD_SECONDS):
                _stats["auto_released"] += 1
                logging.warning("clock held for over %.0f s: released automatically", MAX_HOLD_SECONDS)
                _running.set()
        return orig(self, *a, **kw)

    setattr(Home, name, gated)

    from fastapi import Body

    from src.simulator.api import app as app_mod
    from src.simulator.api.routes import get_home

    def clock(payload: dict = Body(...)):
        hold = bool(payload.get("hold"))
        if hold and _running.is_set():
            _held_since[0] = time.time()
            _stats["holds"] += 1
            _running.clear()
        elif not hold and not _running.is_set():
            _stats["held_seconds"] += time.time() - _held_since[0]
            _running.set()
        return {"held": not _running.is_set(), "tick": int(get_home().current_tick), **_stats}

    app_mod.app.add_api_route("/api/__clock__", clock, methods=["POST"])
    return app_mod.app


def main() -> None:
    import uvicorn

    app = _install()
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("SERVER_PORT", "8000")), reload=False, log_level="warning")


if __name__ == "__main__":
    main()
