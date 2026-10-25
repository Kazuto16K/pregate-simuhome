"""Meaning-preserving rewording of benchmark requests, for the parser's robustness test.

An LLM rewrites a request in casual, varied English. Only the WORDING may change, so the benchmark's answer key
(goals, times, contradictions) still applies. `check` rejects a rewording that changed the meaning:
  - every number must survive (minutes, percentages, temperatures, clock times; "twenty" and "20" are the same,
    "80.0%" and "80%" are the same);
  - every room named in the original must still be named;
  - every device phrase "<device word> <number>" must survive (the device's identity).
Invalid rewordings are retried; the notebook reports how many requests never got a valid one.
"""

from __future__ import annotations

import collections
import re

from .grounding import _words_to_numbers

SYSTEM = (
    "You rewrite smart-home voice requests the way a real person would casually say them. Keep the meaning EXACTLY: "
    "the same devices (write every device exactly as given, e.g. 'living room light 1', 'dryer 2 in the utility "
    "room'), the same rooms, the same actions and settings, and every number and every clock time. You may change "
    "word order, sentence structure, filler words and the way times are expressed (e.g. '20 minutes from now' -> "
    "'in 20 minutes'; '8 minutes after the previous action' -> '8 minutes after that'; '6 14 PM' -> '6:14 pm'). "
    "Do not add, drop or merge requests. Do not fix or point out contradictions. Output only the rewritten request."
)

ROOMS = ("living room", "kitchen", "bedroom", "bathroom", "utility room", "office", "dining room", "kids room",
         "garage", "hallway", "study", "laundry room", "guest room")

_DEVICE_WORDS = r"(light|lamp|blinds?|fan|purifier|dehumidifier|humidifier|heat pump|air conditioner|ac|refrigerator|" \
                r"fridge|freezer|washer|dryer|dishwasher|vacuum|tv|television|dimmer|window covering|oven|plug|switch)"


def _normal(text: str) -> str:
    # number words -> digits ("twenty" -> "20"), then re-join decimals the tokenizer split ("80 . 0" -> "80.0")
    return re.sub(r"(\d) \. (\d)", lambda m: m.group(1) + "." + m.group(2), _words_to_numbers(text.lower()))


def _numbers(text: str) -> collections.Counter:
    t = _normal(text)
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    return collections.Counter(f"{float(n):g}" for n in nums)   # "80.0" and "80" are the same number


def _devices(text: str) -> collections.Counter:
    t = _normal(text)
    return collections.Counter(f"{m.group(1)} {m.group(2)}" for m in re.finditer(_DEVICE_WORDS + r"\s+(\d+)\b", t))


def check(original: str, reworded: str) -> list[str]:
    """Reasons the rewording is not meaning-preserving ([] = accepted)."""
    why = []
    if not reworded or len(reworded) < 0.4 * len(original):
        return ["empty or much shorter"]
    a, b = _numbers(original), _numbers(reworded)
    if a != b:
        why.append(f"numbers differ: lost {dict(a - b)}, added {dict(b - a)}")
    lo, lr = original.lower(), reworded.lower()
    lost_rooms = [r for r in ROOMS if r in lo and r not in lr]
    if lost_rooms:
        why.append(f"rooms lost: {lost_rooms}")
    lost_dev = _devices(original) - _devices(reworded)
    if lost_dev:
        why.append(f"device phrases lost: {dict(lost_dev)}")
    if lr.strip() == lo.strip():
        why.append("unchanged")
    return why
