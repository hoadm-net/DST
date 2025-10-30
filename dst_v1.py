"""
DST v1: Rule-based baseline using ontology
Phương án 1: Dùng ontology + heuristics đơn giản để theo dõi belief state

Mục tiêu học tập:
- Hiểu rõ input (utterance + lịch sử) và output (slot_values, requested_slots)
- Hiểu các độ đo: JGA, Slot F1, Requested F1
- Chuẩn hóa và log chi tiết để debug

Cách chạy:
    python dst_v1.py --split test
    python dst_v1.py --split dev --log_dir logs
"""
from __future__ import annotations
import argparse
import json
import logging
import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

from mint.dst.data import extract_gold_states, iter_user_turns
from mint.dst.evaluation import compute_jga, compute_requested_f1, compute_slot_f1
from mint.dst.normalize import normalize_text, normalize_value
from mint.dst.ontology import Ontology


class DSTV1:
    """
    Rule-based baseline using a small ontology.
    - Closed slots: match ontology values in USER utterance (normalized) and carry over.
    - Open slots: simple regex/keyword extraction (time, numbers, from/to cities).
    - Requested slots: keyword matching; domain resolved by recent activity.
    """

    def __init__(self, ontology: Ontology | None = None) -> None:
        self.ontology = ontology or Ontology()
        self.state: Dict[str, List[str]] = {}
        self.requested: List[str] = []
        self.active_domains: List[str] = []  # track recent domains mentioned

        # simple keyword to requested slot mapping (default domain resolved later)
        self.req_keywords = {
            "address": "address",
            "postcode": "postcode",
            "postal code": "postcode",
            "zip": "postcode",
            "phone": "phone",
            "telephone": "phone",
            "price": "pricerange",
            "price range": "pricerange",
            "reference": "ref",
            "confirmation": "ref",
        }

        # precompiled regex
        self.people_re = re.compile(r"\b(?:for\s+)?(\d{1,2})\s+(?:people|persons|adults)\b")
        self.time_re = re.compile(r"\b(\d{1,2})[:\.](\d{2})\b")
        self.from_re = re.compile(r"\bfrom\s+([a-zA-Z ]{2,})\b")
        self.to_re = re.compile(r"\bto\s+([a-zA-Z ]{2,})\b")

    def reset(self) -> None:
        """Reset state cho hội thoại mới"""
        self.state = {}
        self.requested = []
        self.active_domains = []

    def _add_value(self, slot: str, value: str) -> None:
        """Thêm giá trị vào state, chuẩn hóa và tránh trùng"""
        v = normalize_value(slot, value)
        if not v:
            return
        lst = self.state.setdefault(slot, [])
        if v not in lst:
            lst.append(v)
        # update active domain
        dom = slot.split("-", 1)[0]
        if not self.active_domains or self.active_domains[-1] != dom:
            self.active_domains.append(dom)

    def _set_requested(self, domain: str, slotname: str) -> None:
        """Đánh dấu slot đang được người dùng yêu cầu"""
        slot = f"{domain}-{slotname}"
        if slot not in self.requested:
            self.requested.append(slot)

    def _guess_domain(self, text: str) -> str | None:
        """Đoán domain đang hoạt động dựa trên từ khóa và lịch sử"""
        # Prefer last active domain
        if self.active_domains:
            return self.active_domains[-1]
        # Keyword hints
        if any(k in text for k in ("train", "depart", "arrival", "leave")):
            return "train"
        if any(k in text for k in ("restaurant", "food", "dine", "eat")):
            return "restaurant"
        if "hotel" in text:
            return "hotel"
        return None

    def _extract_closed_slots(self, text: str) -> None:
        """Trích slot đóng bằng cách khớp với ontology values"""
        for slot, values in self.ontology._map.items():  # type: ignore[attr-defined]
            for v in values:
                if not v or v == "dontcare":
                    continue
                if f" {v} " in f" {text} ":
                    self._add_value(slot, v)

    def _extract_time_and_numbers(self, text: str) -> None:
        """Trích thời gian và số lượng bằng regex"""
        # time for train slots
        m = self.time_re.search(text)
        if m:
            tval = f"{int(m.group(1)):02d}:{m.group(2)}"
            # prefer leaveat if present
            self._add_value("train-leaveat", tval)
        # people count
        m2 = self.people_re.search(text)
        if m2:
            self._add_value("train-bookpeople", m2.group(1))

    def _extract_from_to(self, text: str) -> None:
        """Trích điểm đi/đến bằng regex"""
        m_from = self.from_re.search(text)
        if m_from:
            self._add_value("train-departure", m_from.group(1).strip())
        m_to = self.to_re.search(text)
        if m_to:
            self._add_value("train-destination", m_to.group(1).strip())

    def _extract_requested(self, text: str) -> None:
        """Phát hiện requested slots dựa trên từ khóa"""
        dom = self._guess_domain(text) or "restaurant"
        for k, slotname in self.req_keywords.items():
            if k in text:
                self._set_requested(dom, slotname)

    def predict_user_turn(self, user_utterance: str, prev_system_utt: str | None = None) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Dự đoán belief state tại lượt USER hiện tại.
        
        Input:
        - user_utterance: câu nói của người dùng (đã chuẩn hóa sẽ tốt hơn)
        - prev_system_utt: (tùy chọn) câu nói hệ thống ở lượt trước
        
        Output:
        - slot_values: Dict[str, List[str]] - trạng thái đầy đủ hiện tại
        - requested_slots: List[str] - các slot đang được hỏi
        
        Note: Baseline này giữ state qua các lượt (carryover), gọi reset() cho hội thoại mới.
        """
        text = normalize_text(user_utterance)
        # heuristics
        self._extract_closed_slots(text)
        self._extract_time_and_numbers(text)
        self._extract_from_to(text)
        self._extract_requested(text)
        # return copies
        return {k: v[:] for k, v in self.state.items()}, self.requested[:]


def evaluate_split(data_dir: Path, split: str, log_dir: Path) -> None:
    """
    Đánh giá DST baseline trên một split (dev/test)
    Ghi log chi tiết từng turn để debug
    """
    split_path = data_dir / split
    log_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dst_v1_{split}_{ts}.log"
    details_file = log_dir / f"dst_v1_{split}_details_{ts}.jsonl"
    summary_file = log_dir / f"dst_v1_{split}_summary_{ts}.json"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    logging.info("="*60)
    logging.info("DST V1 - Rule-based Baseline")
    logging.info("="*60)
    logging.info(f"Split: {split}")
    logging.info(f"Data: {split_path}")
    
    files = sorted(glob(str(split_path / "*.json")))
    logging.info(f"Tìm thấy {len(files)} file dữ liệu")
    
    pred_states_all = []
    gold_states_all = []
    pred_requests_all = []
    gold_requests_all = []
    
    total_dialogues = 0
    total_turns = 0
    
    with details_file.open("w") as fw:
        for fp in files:
            with open(fp) as f:
                dialogues = json.load(f)
            
            for dialogue in dialogues:
                total_dialogues += 1
                dialogue_id = dialogue.get("dialogue_id")
                
                # Khởi tạo model mới cho mỗi dialogue
                dst = DSTV1()
                
                # Lấy ground truth
                gold_states, gold_requests = extract_gold_states(dialogue)
                
                # Dự đoán từng turn
                turn_idx = 0
                for turn in iter_user_turns(dialogue):
                    total_turns += 1
                    utterance = turn.get("utterance", "")
                    
                    # Predict
                    pred_state, pred_req = dst.predict_user_turn(utterance)
                    
                    pred_states_all.append(pred_state)
                    pred_requests_all.append(pred_req)
                    
                    # Log chi tiết để debug
                    if turn_idx < len(gold_states):
                        gold_state = gold_states[turn_idx]
                        gold_req = gold_requests[turn_idx] if turn_idx < len(gold_requests) else []
                        
                        detail = {
                            "dialogue_id": dialogue_id,
                            "turn_id": turn.get("turn_id"),
                            "utterance": utterance,
                            "gold_state": gold_state,
                            "pred_state": pred_state,
                            "gold_requested": gold_req,
                            "pred_requested": pred_req,
                            "match": gold_state == pred_state and gold_req == pred_req
                        }
                        fw.write(json.dumps(detail, ensure_ascii=False, indent=2) + "\n")
                    
                    turn_idx += 1
                
                gold_states_all.extend(gold_states)
                gold_requests_all.extend(gold_requests)
    
    # Tính metrics
    logging.info("="*60)
    logging.info("Đang tính toán metrics...")
    
    jga = compute_jga(gold_states_all, pred_states_all)
    slot_p, slot_r, slot_f1 = compute_slot_f1(gold_states_all, pred_states_all)
    req_p, req_r, req_f1 = compute_requested_f1(gold_requests_all, pred_requests_all)
    
    # Log kết quả
    logging.info("="*60)
    logging.info("KẾT QUẢ ĐÁNH GIÁ")
    logging.info("="*60)
    logging.info(f"Tổng số dialogues: {total_dialogues}")
    logging.info(f"Tổng số USER turns: {total_turns}")
    logging.info("-"*60)
    logging.info(f"Joint Goal Accuracy (JGA): {jga:.4f}")
    logging.info("-"*60)
    logging.info(f"Slot F1:")
    logging.info(f"  Precision: {slot_p:.4f}")
    logging.info(f"  Recall:    {slot_r:.4f}")
    logging.info(f"  F1:        {slot_f1:.4f}")
    logging.info("-"*60)
    logging.info(f"Requested Slots F1:")
    logging.info(f"  Precision: {req_p:.4f}")
    logging.info(f"  Recall:    {req_r:.4f}")
    logging.info(f"  F1:        {req_f1:.4f}")
    logging.info("="*60)
    logging.info(f"Log file: {log_file}")
    logging.info(f"Details: {details_file}")
    logging.info(f"Summary: {summary_file}")
    
    # Lưu summary
    summary = {
        "split": split,
        "timestamp": ts,
        "dialogues": total_dialogues,
        "user_turns": total_turns,
        "metrics": {
            "JGA": jga,
            "slot_f1": {
                "precision": slot_p,
                "recall": slot_r,
                "f1": slot_f1
            },
            "requested_f1": {
                "precision": req_p,
                "recall": req_r,
                "f1": req_f1
            }
        }
    }
    
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="DST V1: Rule-based baseline với ontology"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("datasets/MultiWOZ"),
        help="Thư mục chứa dữ liệu MultiWOZ"
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="Split để đánh giá (dev hoặc test)"
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("logs"),
        help="Thư mục lưu log"
    )
    
    args = parser.parse_args()
    evaluate_split(args.data_dir, args.split, args.log_dir)


if __name__ == "__main__":
    main()
