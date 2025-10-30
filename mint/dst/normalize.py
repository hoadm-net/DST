import re
from typing import Dict

# Simple synonyms/canonicalization mapping
_SYNONYMS: Dict[str, str] = {
    "center": "centre",
    "city centre": "centre",
    "dont care": "dontcare",
    "don't care": "dontcare",
    "do not care": "dontcare",
}

_DAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
}

_TIME_RE = re.compile(r"\b(\d{1,2})[:\.](\d{2})\b")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # apply simple synonyms
    for k, v in _SYNONYMS.items():
        s = s.replace(k, v)
    return s

def _normalize_time(value: str) -> str:
    m = _TIME_RE.search(value)
    if not m:
        return value
    h, mnt = int(m.group(1)), m.group(2)
    # zero-pad hour and bound to 00-23 (best effort)
    h = max(0, min(23, h))
    return f"{h:02d}:{mnt}"

def normalize_value(slot: str, value: str) -> str:
    v = normalize_text(value)
    if not v:
        return v
    # time-like slots
    if any(key in slot for key in ("leaveat", "arriveby", "time")):
        v = _normalize_time(v)
    # day of week
    if "day" in slot and v in _DAYS:
        return v
    # boolean slots canonicalization
    if v in {"yes", "true"}:
        return "yes"
    if v in {"no", "false"}:
        return "no"
    return v
