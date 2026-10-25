"""The clock-hold server (pausable_sim.py): the simulated clock stops while held and runs otherwise.

Run: python tests/test_clock_hold.py [path/to/SimuHome]
Starts one real SimuHome server through pausable_sim.py (needs SimuHome's requirements installed).
"""

from __future__ import annotations

import os
import sys
import time

import _paths


def main(simuhome_dir: str) -> int:
    simuhome_dir = _paths.setup(simuhome_dir)
    import requests

    from irlib import parallel_runner as pr

    servers = pr.start_simulators(1, simuhome_dir, pausable=True)
    port = servers[0][0]
    base = f"http://127.0.0.1:{port}/api"

    def now_seconds() -> float:
        r = requests.get(f"{base}/time", timeout=10).json()
        data = r.get("data") or r
        return float(data.get("tick", 0)) * 0.1 if "tick" in data else _clock_seconds(data)

    def _clock_seconds(data) -> float:
        from datetime import datetime
        return datetime.strptime(data["now"], "%Y-%m-%d %H:%M:%S").timestamp()

    failures = 0
    try:
        t0 = now_seconds()
        time.sleep(3.0)
        t1 = now_seconds()
        running = t1 - t0
        print(f"clock running: +{running:.1f} s simulated in 3 s")
        failures += not (2.0 <= running <= 4.5)

        pr._TLS.port = port
        with pr.llm_wait():  # what the notebook's LLM shim does around every request
            time.sleep(3.0)
        t2 = now_seconds()
        held = t2 - t1
        print(f"clock held for 3 s: +{held:.1f} s simulated (should be ~0)")
        failures += not (held <= 1.2)

        time.sleep(2.0)
        t3 = now_seconds()
        print(f"released again: +{t3 - t2:.1f} s simulated in 2 s")
        failures += not (1.0 <= t3 - t2 <= 3.5)
        print("hold stats:", pr.CLOCK)
        failures += pr.CLOCK["errors"] != 0
    finally:
        pr._TLS.port = None
        pr.stop_simulators(servers)
    print("FAILED" if failures else "OK: the clock stops while held and runs otherwise")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(_paths.PROJECT, "..", "SimuHome")))
