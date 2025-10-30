from __future__ import annotations
from typing import Dict, Iterable, List, Tuple

# Utilities to work with this repo's MultiWOZ v2.2-style files that include per-turn frames


def aggregate_user_turn_state(turn: Dict) -> Tuple[str, Dict[str, List[str]], List[str]]:
    """
    Given a USER turn (with frames), aggregate slot_values across services and requested_slots.
    Returns: (utterance, slot_values, requested_slots)
    """
    utterance = turn.get("utterance", "")
    slot_values: Dict[str, List[str]] = {}
    requested: List[str] = []
    for fr in turn.get("frames", []):
        st = fr.get("state", {})
        # slot_values
        for slot, vals in st.get("slot_values", {}).items():
            slot_values.setdefault(slot, [])
            for v in vals:
                if v not in slot_values[slot]:
                    slot_values[slot].append(v)
        # requested_slots
        for r in st.get("requested_slots", []):
            if r not in requested:
                requested.append(r)
    return utterance, slot_values, requested


def iter_user_turns(dialogue: Dict) -> Iterable[Dict]:
    for t in dialogue.get("turns", []):
        if t.get("speaker") == "USER":
            yield t


def extract_gold_states(dialogue: Dict) -> Tuple[List[Dict[str, List[str]]], List[List[str]]]:
    gold_states: List[Dict[str, List[str]]] = []
    gold_requests: List[List[str]] = []
    for t in iter_user_turns(dialogue):
        _, slot_values, requested = aggregate_user_turn_state(t)
        gold_states.append(slot_values)
        gold_requests.append(requested)
    return gold_states, gold_requests
