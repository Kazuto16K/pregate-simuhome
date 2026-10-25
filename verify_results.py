"""Recompute every reported number from the stored data and check it against the expected value.

Run:  python verify_results.py [path/to/SimuHome]
Needs the SimuHome checkout only for the benchmark files (parser table). Prints each check as OK or MISMATCH.

Data: results/qwen2.5-14b/shard0 + shard1 (SimuHome's per-episode records, the gate's per-episode telemetry and the
run logs of every session) and data/parser/parse_cache_v3_cv.json (5-fold parses).
Scoring: "strict" counts episodes that ended in an infrastructure error (score -1) as failures.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import _paths  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "qwen2.5-14b")
QTS = ("qt4-1", "qt4-2", "qt4-3")
failures = []


def expect(name, got, want):
    ok = got == want if not isinstance(want, float) else abs(got - want) <= 1e-6 * max(1.0, abs(want)) * 100
    print(f"  {'OK      ' if ok else 'MISMATCH'} {name}: {got}" + ("" if ok else f"   (expected {want})"))
    if not ok:
        failures.append(name)


def load():
    recs, tel, logs = {}, {}, []
    for shard in ("shard0", "shard1"):
        for f in glob.glob(os.path.join(RESULTS, shard, "SimuHome", "experiments", "a2_*", "*", "qt4-*.json")):
            run = f.replace("\\", "/").split("/experiments/")[1].split("/")[0]
            arm = run[3:].split("_qt4")[0]
            r = json.load(open(f, encoding="utf-8"))
            key = (arm, r["query_type"], r["case"], r["seed"])
            assert key not in recs, f"duplicate episode {key}"
            recs[key] = r
        for f in glob.glob(os.path.join(RESULTS, shard, "gate_telemetry_*.json")):
            for t in json.load(open(f, encoding="utf-8")):
                tel[t["episode"]] = t
        for f in glob.glob(os.path.join(RESULTS, shard, "e1_run_log_*.json")):
            logs += json.load(open(f, encoding="utf-8"))
    return recs, tel, logs


def main(simuhome_dir: str) -> int:
    simuhome_dir = _paths.setup(simuhome_dir)
    from irlib.analysis import mcnemar_exact
    from irlib.extract_ir_from_episode import extract_dataset
    from irlib.parse_eval import score, table

    recs, tel, logs = load()
    sc = lambda r: (r.get("evaluation_result") or {}).get("score")
    strict = {k: 1 if sc(r) == 1 else 0 for k, r in recs.items()}

    print("1. setup of every session (run logs)")
    expect("agent/judge model", sorted({e["model"] for e in logs}), ["simuhome-agent"])
    expect("gate version", sorted({e["gate_version"] for e in logs}), ["2.2"])
    expect("clock held during LLM calls", sorted({e["clock_hold"] for e in logs}), [True])
    expect("clock-hold errors (last entry of each session)", sum(e.get("clock", {}).get("errors", 0) for e in logs), 0)
    expect("episode records", len(recs), 1044)

    print("2. same episodes in every arm")
    for qt in QTS:
        for case, arms in (("feasible", ("baseline", "selfcheck", "full")), ("infeasible", ("baseline", "selfcheck", "full", "full_answer"))):
            sets = [{k[3] for k in recs if k[:3] == (a, qt, case)} for a in arms]
            expect(f"{qt} {case}: identical episode sets", all(s == sets[0] for s in sets), True)

    print("3. no parser leakage (each episode parsed by the fold that held it out)")
    cache = json.load(open(os.path.join(ROOT, "data", "parser", "parse_cache_v3_cv.json"), encoding="utf-8"))
    bench = os.path.join(simuhome_dir, "data", "benchmark")
    wrong = 0
    for (qt, case, seed) in {k[1:] for k in recs}:
        q = json.load(open(os.path.join(bench, f"{qt}_{case}_seed_{seed}.json"), encoding="utf-8"))["query"]
        wrong += cache[q].get("fold") != seed % 5
    expect("episodes parsed by the wrong fold", wrong, 0)

    print("4. main table (strict)")
    want = {("feasible", "baseline"): (0, 2, 0), ("feasible", "selfcheck"): (0, 1, 0), ("feasible", "full"): (6, 9, 0),
            ("infeasible", "baseline"): (3, 5, 1), ("infeasible", "selfcheck"): (5, 11, 3),
            ("infeasible", "full"): (49, 50, 30), ("infeasible", "full_answer"): (49, 50, 31)}
    for (case, arm), w in want.items():
        got = tuple(sum(v for k, v in strict.items() if k[:3] == (arm, qt, case)) for qt in QTS)
        expect(f"{case:10s} {arm:12s} correct per sub-type", got, w)

    print("5. paired McNemar tests (pooled over sub-types)")
    def pair(a, b, case):
        ks = [(k[1], k[3]) for k in strict if k[0] == a and k[2] == case]
        oa = sum(strict[(a, q, case, s)] > strict[(b, q, case, s)] for q, s in ks)
        ob = sum(strict[(a, q, case, s)] < strict[(b, q, case, s)] for q, s in ks)
        return oa, ob, mcnemar_exact(oa, ob)
    for (a, b, case, w) in (("full", "baseline", "feasible", (14, 1)), ("full", "selfcheck", "feasible", (15, 1)),
                            ("full", "baseline", "infeasible", (121, 1)), ("full", "selfcheck", "infeasible", (110, 0)),
                            ("full_answer", "baseline", "infeasible", (121, 0)), ("full_answer", "selfcheck", "infeasible", (111, 0))):
        oa, ob, p = pair(a, b, case)
        expect(f"{case:10s} {a} vs {b}: discordant pairs", (oa, ob), w)
        print(f"           p = {p:.3g}")
    expect("possible requests, gate vs agent alone: p < 0.001", pair("full", "baseline", "feasible")[2] < 0.001, True)
    expect("impossible requests, gate vs both baselines: p < 1e-30",
           max(pair("full", b, "infeasible")[2] for b in ("baseline", "selfcheck")) < 1e-30, True)

    print("6. goals met on possible requests, QT4-1 (episodes without infrastructure error)")
    for arm, w in (("baseline", "19/148"), ("selfcheck", "15/142"), ("full", "56/138")):
        ev = [recs[k]["evaluation_result"] for k in recs if k[:3] == (arm, "qt4-1", "feasible") and sc(recs[k]) in (0, 1)]
        expect(f"{arm}", f"{sum(sum(g['satisfied'] for g in e['goals_eval']) for e in ev)}/{sum(len(e['goals_eval']) for e in ev)}", w)

    print("7. large-model calls per impossible request in answer mode")
    for qt, w in (("qt4-1", 0.1), ("qt4-2", 0.0)):
        ks = [k for k in recs if k[:3] == ("full_answer", qt, "infeasible") and sc(recs[k]) in (0, 1)]
        expect(qt, round(sum(len(recs[k].get("raw_responses") or []) for k in ks) / len(ks), 1), w)

    print("8. judge audit: impossible requests judged correct although the agent scheduled actions")
    def scheduled(r):
        return any(s.get("action") == "schedule_workflow" and isinstance(s.get("observation"), dict)
                   and (s["observation"].get("status") or {}).get("code") == 200 for s in r["steps"])
    for arm, w in (("baseline", 2), ("selfcheck", 9), ("full", 129), ("full_answer", 129)):
        ks = [k for k in recs if k[0] == arm and k[2] == "infeasible"]
        expect(f"{arm}: correct without such cases", sum(sc(recs[k]) == 1 and not scheduled(recs[k]) for k in ks), w)
    ans = [t for t in tel.values() if t["arm"] == "full_answer"]
    wrong_ans = sum(1 for k, r in recs.items() if k[0] == "full_answer" and sc(r) != 1 and tel.get(
        f"a2_full_answer_{k[1].replace('-', '_')}_infeasible_ch_g22/{k[1]}_infeasible_seed_{k[3]}", {}).get("answered_by_gate"))
    expect("answer mode: episodes answered by the gate", sum(t["answered_by_gate"] for t in ans), 129)
    expect("answer mode: gate answers judged wrong", wrong_ans, 0)

    print("9. mechanism and safety on possible requests (gate telemetry)")
    wins = [k for k in strict if k[0] == "full" and k[2] == "feasible" and strict[k] == 1 and strict[("baseline",) + k[1:]] == 0]
    dry = sum(bool(tel.get(f"a2_full_{k[1].replace('-', '_')}_feasible_ch_g22/{k[1]}_feasible_seed_{k[3]}", {}).get("dry_run_rejections")) for k in wins)
    expect("possible requests completed only with the gate", len(wins), 14)
    expect("... of which after a failing-step refusal with the real command", dry, 13)
    logged = [t for k, t in tel.items() if t["arm"] == "full" and "_feasible" in k]
    expect("logged possible requests wrongly called impossible", f"{sum(t['preflight'] == 'unsat' for t in logged)} of {len(logged)}", "0 of 131")

    print("10. small parser, 5-fold, scored as the gate uses it")
    stats, _ = score({k: v for k, v in cache.items() if k != "_meta"}, extract_dataset(bench), bench)
    rows = {r["qt"]: r for r in table(stats)}
    for qt, w in (("qt4-1", ("49/50", "49/50", "1/148")), ("qt4-2", ("49/50", "50/50", "0/49")), ("qt4-3", ("50/50", "30/36", "0/50"))):
        r = rows[qt]
        expect(f"{qt}: possible judged possible / contradictions proven / correct goals refused",
               (r["possible_ok"], r["contradictions_proven"], r["goals_refused"]), w)

    print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} MISMATCH(ES): {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SIMUHOME_DIR", os.path.join(ROOT, "..", "SimuHome"))))
