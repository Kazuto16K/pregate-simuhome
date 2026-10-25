"""Re-simulate the agent's accepted actions of every FAILED possible-request episode (QT4-1/2) and, for each failed goal,
report the actual vs expected device state and the workflow errors. Run: python tests/diagnose_failures.py <results dir>..."""
import json, sys, glob, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
SH = _paths.setup(os.environ.get("SIMUHOME_DIR", os.path.join(_paths.PROJECT, "..", "SimuHome")))
from src.pipelines.episode_generation.core.home_augmenter import _initialize_home_locally
BENCH = os.path.join(SH, "data", "benchmark")
def obs(s):
    o = s.get("observation")
    if isinstance(o, str):
        try: o = json.loads(o)
        except Exception: return {}
    return o or {}
def code(s): return ((obs(s).get("status")) or {}).get("code")
out = collections.Counter(); lines = []
for root in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(root, "SimuHome", "experiments", "*qt4_[12]_feasible*", "*", "qt4-*.json"))):
        run = f.replace("\\", "/").split("/experiments/")[1].split("/")[0]
        r = json.load(open(f, encoding="utf-8")); ev = r.get("evaluation_result") or {}
        if ev.get("score") != 0: continue
        name = os.path.basename(f); ep = json.load(open(os.path.join(BENCH, name), encoding="utf-8"))
        home, _ = _initialize_home_locally(ep["initial_home_config"])
        wfs = []
        for st in r.get("steps") or []:
            a, args = st.get("action"), st.get("action_input") or {}
            if not isinstance(args, dict) or code(st) not in (200, None): continue
            try:
                if a == "execute_command":
                    res = home._execute_command(args["device_id"], args["endpoint_id"], args["cluster_id"], args["command_id"], args.get("args") or {})
                    if not res.success: wfs.append(("immediate", args.get("command_id"), res.error_message))
                elif a == "write_attribute":
                    res = home._write_attribute(args["device_id"], args["endpoint_id"], args["cluster_id"], args["attribute_id"], args["value"])
                    if not res.success: wfs.append(("immediate", args.get("attribute_id"), res.error_message))
                elif a == "schedule_workflow" and code(st) == 200:
                    res = home.schedule_workflow(args["start_time"], args["steps"])
                    if res.success: wfs.append(("wf", res.data["workflow_id"], args))
            except Exception as e:
                wfs.append(("immediate", a, f"{type(e).__name__}: {e}"))
        goals = ep["eval"]["goals"]
        for g in ev.get("goals_eval") or []:
            if g["satisfied"]: continue
            spec = goals[g["goal_index"]]; at = spec["when"]["at_tick"] + spec["when"]["tolerance_ticks"]
            home._fast_forward_to(at)
            exp_act = []
            for t in spec["targets"]:
                for asr in t["asserts"]:
                    e_, c_, a_ = asr["attribute"].split(".", 2)
                    try: v = home._get_attribute(t["device_id"], int(e_), c_, a_).data
                    except Exception as e: v = f"ERR {e}"
                    if isinstance(v, dict) and "value" in v: v = v["value"]
                    exp_act.append(f"{t['device_id']}.{c_}.{a_}: want {asr['value']!r} got {v!r}")
            errs = []
            for w in wfs:
                if w[0] == "immediate": errs.append(f"immediate {w[1]} failed: {w[2]}")
                else:
                    stt = home.get_workflow_status(w[1])
                    d = stt.data if stt.success else {}
                    if d.get("status") == "failed" or d.get("error"):
                        errs.append(f"wf@{w[2]['start_time'][11:]} failed: {str(d.get('error'))[:70]}")
            lines.append(f"{run[3:]:32s} {name[:-5]:24s} g{g['goal_index']} t={at/600:.1f}m | " + " ; ".join(exp_act) + (" || " + " | ".join(errs) if errs else ""))
print("\n".join(lines))
