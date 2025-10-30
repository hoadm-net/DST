from __future__ import annotations
from typing import Dict, Iterable, List, Set, Tuple
from .normalize import normalize_value

State = Dict[str, List[str]]  # domain-slot -> list of values


def _flatten_slot_values(slot_values: Dict[str, List[str]]) -> Set[Tuple[str, str]]:
    items: Set[Tuple[str, str]] = set()
    for slot, values in slot_values.items():
        for v in values:
            items.add((slot, normalize_value(slot, v)))
    return items


def _normalize_state(slot_values: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for slot, values in slot_values.items():
        out.setdefault(slot, set()).update(normalize_value(slot, v) for v in values)
    return out


def compute_jga(
    gold_states: Iterable[Dict[str, List[str]]],
    pred_states: Iterable[Dict[str, List[str]]],
) -> float:
    total = 0
    correct = 0
    for g, p in zip(gold_states, pred_states):
        total += 1
        g_norm = _normalize_state(g)
        p_norm = _normalize_state(p)
        if g_norm == p_norm:
            correct += 1
    return correct / total if total else 0.0


def compute_slot_f1(
    gold_states: Iterable[Dict[str, List[str]]],
    pred_states: Iterable[Dict[str, List[str]]],
) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for g, p in zip(gold_states, pred_states):
        g_set = _flatten_slot_values(g)
        p_set = _flatten_slot_values(p)
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_requested_f1(
    gold_requests: Iterable[List[str]],
    pred_requests: Iterable[List[str]],
) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for g, p in zip(gold_requests, pred_requests):
        g_set = set(g)
        p_set = set(p)
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
