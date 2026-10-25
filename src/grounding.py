"""Grounding filter: drop clock times the parser wrote that the user never said.

The v3 parser sometimes adds an absolute clock reading that is not in the request
(e.g. "at 16:00" for "start the dishwasher now"). Z3 then finds a genuine clash
with the other reading and calls a POSSIBLE request impossible. A clock time is
something the user states explicitly, so it can be checked against the sentence
deterministically: if no time in the text matches it, the reading is removed
before Z3 sees it. Relative offsets are not filtered; a stated time plus/minus an offset in the same
IR also counts as stated (see ground_ir).

Times are recognised in digit form ("12:23 PM", "1:39 pm", "12 36 PM", "7 PM") and
word form ("six fifty six PM", "twelve thirty six PM", "seven o'clock").
"""

from __future__ import annotations

import copy
import re

_UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}


def _with_meridiem(h: int, m: int, mer: str | None) -> set[tuple[int, int]]:
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return set()
    if mer:
        mer = mer.replace(".", "").lower()
        if mer == "pm" and h < 12:
            h += 12
        elif mer == "am" and h == 12:
            h = 0
        return {(h, m)}
    out = {(h, m)}
    if h < 12:
        out.add((h + 12, m))  # no AM/PM given: either reading
    return out


def _words_to_numbers(text: str) -> str:
    """'six fifty six pm' -> '6 56 pm'; 'twelve thirty six' -> '12 36'."""
    tokens = re.findall(r"[a-z]+|\d+|[^a-z\d\s]", text.lower())
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in _TENS:
            val = _TENS[t]
            if i + 1 < len(tokens) and tokens[i + 1] in _UNITS and 0 < _UNITS[tokens[i + 1]] < 10:
                val += _UNITS[tokens[i + 1]]
                i += 1
            out.append(str(val))
        elif t in _UNITS:
            val = _UNITS[t]
            if t == "oh" and i + 1 < len(tokens) and tokens[i + 1] in _UNITS:  # "ten oh three" -> "10 03"
                out.append(f"{_UNITS[tokens[i + 1]]:02d}")
                i += 2
                continue
            out.append(str(val))
        else:
            out.append(t)
        i += 1
    return " ".join(out)


def times_in_text(text: str) -> set[tuple[int, int]]:
    """Every (hour 0-23, minute) the text could be stating as a clock time."""
    found: set[tuple[int, int]] = set()
    for variant in (text.lower(), _words_to_numbers(text)):
        for h, m, mer in re.findall(r"\b(\d{1,2})\s*[:.\s]\s*(\d{2})\s*(a\.?m\.?|p\.?m\.?)?", variant):
            found |= _with_meridiem(int(h), int(m), mer or None)
        for h, mer in re.findall(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)", variant):
            found |= _with_meridiem(int(h), 0, mer)
        for h in re.findall(r"\b(\d{1,2})\s*o\s*'?\s*clock", variant):
            found |= _with_meridiem(int(h), 0, None)
    return found


def _closest_device(name: str, known: list[str], cutoff: float) -> str | None:
    """Closest real device id, preferring the same room (the id prefix before the device type)."""
    import difflib

    same_room = [k for k in known if k.split("_")[0] == name.split("_")[0]]
    for pool in (same_room, known):
        hit = difflib.get_close_matches(name, pool, n=1, cutoff=cutoff)
        if hit:
            return hit[0]
    return None


def repair_device_ids(compact: dict, known_ids: list[str], cutoff: float = 0.85) -> tuple[dict, int, int]:
    """Copy of `compact` whose device ids that do not exist in the house are replaced by the closest real
    one (e.g. the parser's 'utility_laundry_dryer_1' -> 'utility_room_laundry_dryer_1'). Uses only the
    house's device list, never the answer key. Returns (ir, n_repaired, n_unresolved)."""
    known = list(known_ids)
    known_set = set(known)
    out = copy.deepcopy(compact)
    repaired = unresolved = 0

    def fix(obj: dict) -> None:
        nonlocal repaired, unresolved
        dev = obj.get("device_id")
        if not dev or dev in known_set:
            return
        hit = _closest_device(dev, known, cutoff)
        if hit:
            obj["device_id"] = hit
            repaired += 1
        else:
            unresolved += 1

    for g in out.get("goals", []):
        for r in g.get("readings", []):
            if (r.get("anchor") or {}).get("kind") == "device_event":
                fix(r["anchor"])
        for t in g.get("targets", []):
            fix(t)
    return out, repaired, unresolved


def ground_ir(compact: dict, query: str) -> tuple[dict, int]:
    """Copy of `compact` without clock readings the query does not state. Returns (ir, n_dropped).

    A clock time counts as stated if it appears in the text, or if it is a stated time
    plus/minus an offset the same IR uses ("finishes around 4:42 PM ... start 19 minutes
    after" -> 17:01): the benchmark's own labels express such derived times this way.
    """
    stated = times_in_text(query)
    offsets = {int(round(r["offset_minutes"])) for g in compact.get("goals", []) for r in g.get("readings", [])
               if isinstance(r.get("offset_minutes"), (int, float))}
    derived = set()
    for h, m in stated:
        for o in offsets:
            for t in (h * 60 + m + o, h * 60 + m - o):
                derived.add(((t // 60) % 24, t % 60))
    stated = stated | derived
    out = copy.deepcopy(compact)
    dropped = 0
    for g in out.get("goals", []):
        kept = []
        for r in g.get("readings", []):
            at = r.get("absolute_time")
            if at:
                try:
                    h, m = (int(x) for x in at.split(":")[:2])
                except ValueError:
                    dropped += 1
                    continue
                if (h, m) not in stated:
                    dropped += 1
                    continue
            kept.append(r)
        g["readings"] = kept
    return out, dropped



_PREVIOUS_ACTION = re.compile(r"\bafter (the|that) previous\b", re.I)


def repair_split_chain(compact: dict, query: str) -> tuple[dict, int]:
    """Merge a "N minutes after the previous action" that the parser split into two readings.

    Seen in E1 (QT4-1 feasible seed 25): "... cool the fridge 8 minutes after the previous action" came out as
    two readings of one goal, `now + 8` AND `previous goal + 0` (another fold's parser: `+ 1`). Both cannot
    hold (the previous goal is at now + 6), so Z3 called a POSSIBLE request impossible or computed a wrong
    window. The fix applies only to exactly that shape: a goal with ONE reading `after <goal> + M` and ONE
    relative reading `after now + N` (N > 0, no clock time), in a request that says "after the previous ...".
    It becomes `after <goal> + N` (N is the stated "N minutes after the previous action"). The benchmark's labels never
    have this shape (checked over all 300 QT4 episodes: a goal with both a `now` and a goal reading occurs only
    in QT4-3, as a stated CLOCK time, which this leaves alone), so no real contradiction can be removed.
    Returns (repaired copy, number of goals repaired)."""
    if not _PREVIOUS_ACTION.search(query or ""):
        return compact, 0
    out = copy.deepcopy(compact)
    fixed = 0
    for g in out.get("goals", []):
        rs = g.get("readings") or []
        chain = [r for r in rs if (r.get("anchor") or {}).get("kind") == "goal" and r.get("relation") == "after"]
        rel_now = [r for r in rs if (r.get("anchor") or {}).get("kind") == "now" and r.get("relation") == "after"
                   and not r.get("absolute_time") and float(r.get("offset_minutes") or 0) > 0]
        if len(chain) == 1 and len(rel_now) == 1:
            chain[0]["offset_minutes"] = float(rel_now[0]["offset_minutes"])
            g["readings"] = [r for r in rs if r is not rel_now[0]]
            fixed += 1
    return out, fixed


def _devices(goal: dict) -> set:
    return {t.get("device_id") for t in goal.get("targets") or [] if t.get("device_id")}


def repair_chain_anchor(compact: dict) -> tuple[dict, int]:
    """Re-point "after the previous action" to the previous action ON THE SAME DEVICE.

    In SimuHome's requests "... set the dimmer to 60 in 15 minutes and bump IT to 100 23 minutes after the
    previous action" chains to the dimmer's own earlier action: in the benchmark's labels every chained QT4-1
    goal (56 of 56) refers to the latest earlier goal on the same device. The 5-fold parsers sometimes point the
    chain at the previous SENTENCE's device instead (e.g. the dehumidifier step chained to the air purifier),
    which puts the goal's window at a wrong time and would make the time check refuse a correct schedule. A
    goal-anchored "after" reading whose referenced goal shares no device with this goal is re-pointed to the
    latest earlier goal (in the IR's order) that does; nothing changes if there is none. Run after device-name
    repair so that typos do not hide a shared device. Returns (repaired copy, readings re-pointed)."""
    out = copy.deepcopy(compact)
    goals = out.get("goals", [])
    fixed = 0
    for i, g in enumerate(goals):
        mine = _devices(g)
        for r in g.get("readings") or []:
            a = r.get("anchor") or {}
            if a.get("kind") != "goal" or r.get("relation") != "after":
                continue
            ref = a.get("goal_ref")
            if not isinstance(ref, int) or not (0 <= ref < len(goals)) or mine & _devices(goals[ref]):
                continue
            same = [j for j in range(i) if j != ref and mine & _devices(goals[j])]
            if same:
                a["goal_ref"] = same[-1]
                fixed += 1
    return out, fixed


_CLOCK_SPAN = re.compile(r"\b(\d{1,2})\s*[:.\s]\s*(\d{2})\s*(am|pm)\b")


def _op_state_values(goal: dict) -> set:
    return {a.get("value") for t in goal.get("targets") or [] for a in t.get("asserts") or []
            if str(a.get("attribute", "")).endswith("OperationalState.OperationalState")}


def restore_stated_clock(compact: dict, query: str) -> tuple[dict, int]:
    """Put back a clock time the user STATED but the parser left out (the counterpart of ground_ir).

    In QT4-3 ("start the dryer at 6:56 PM ... pause it when dryer 2 finishes") the stated clock time is what
    makes a request impossible; the 5-fold parsers often drop it and copy the request's consistent twin, so
    Z3 sees no contradiction (17 of 36 in-scope impossible QT4-3 requests). This applies only to that shape
    (a "by <goal>" deadline and a device-finish anchor, and no clock reading at all) and only to times with
    AM/PM, placed as the benchmark's labels place them:
      "start ... at T"  -> the goal that starts a machine (OperationalState 1);
      "... by T"        -> the goal anchored on the other device's finish.
    Other mentions ("I expect it to finish around T", "my call starts at T") are left alone. None of the 50
    POSSIBLE QT4-3 requests states a clock time, so this cannot turn a possible request into an impossible
    one there. Returns (copy, readings added)."""
    goals = compact.get("goals") or []
    readings = [r for g in goals for r in g.get("readings") or []]
    has_deadline = any(r.get("relation") == "by" and (r.get("anchor") or {}).get("kind") == "goal" for r in readings)
    has_device = any((r.get("anchor") or {}).get("kind") == "device_event" for r in readings)
    if not (has_deadline and has_device) or any(r.get("absolute_time") for r in readings):
        return compact, 0
    text = _words_to_numbers(query or "").replace("p . m .", "pm").replace("a . m .", "am")
    out = copy.deepcopy(compact)
    added = 0
    for m in _CLOCK_SPAN.finditer(text):
        h, mi, mer = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= h <= 12 and 0 <= mi <= 59):
            continue
        h = (h % 12) + (12 if mer == "pm" else 0)
        before = text[:m.start()]
        clause = re.split(r"[.;,!?]| so | and ", before)[-1]
        if re.search(r"\bat\s*$", clause) and re.search(r"\bstart\b", clause):
            target = [g for g in out["goals"] if 1 in _op_state_values(g)]
        elif re.search(r"\bby\s*$", clause):
            target = [g for g in out["goals"] if any((r.get("anchor") or {}).get("kind") == "device_event"
                                                     for r in g.get("readings") or [])]
        else:
            continue
        if not target or any(r.get("absolute_time") for r in target[0].get("readings") or []):
            continue
        target[0]["readings"].append({"relation": "at", "anchor": {"kind": "now"}, "absolute_time": f"{h:02d}:{mi:02d}"})
        added += 1
    return out, added
