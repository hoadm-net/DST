"""
DST v2: Per-slot với DistilBERT/BERT
Phương án 2: Learning-based approach

Kiến trúc:
- Closed slots: classification (softmax qua giá trị + none + dontcare)
- Open slots: span extraction (dự đoán start/end token)
- Requested slots: multi-label classification

Mục tiêu:
- Hiểu cách áp dụng BERT cho DST
- Baseline learning-based để so sánh với v1 và chuẩn bị cho GNN (v3)
- Log chi tiết quá trình training và evaluation

Cách chạy:
    # Training
    python dst_v2.py --mode train --epochs 3 --batch_size 8
    
    # Evaluation
    python dst_v2.py --mode eval --split test --checkpoint logs/best_model.pt
"""
from __future__ import annotations
import argparse
import json
import logging
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from mint.dst.data import extract_gold_states, iter_user_turns
from mint.dst.evaluation import compute_jga, compute_requested_f1, compute_slot_f1
from mint.dst.normalize import normalize_text, normalize_value
from mint.dst.ontology import Ontology, default_ontology
from mint.dst.model import SlotClassifier, SpanExtractor, RequestedSlotsClassifier


class DSTDataset(Dataset):
    """
    Dataset cho DST với per-slot approach
    Mỗi sample là một USER turn với context
    """
    def __init__(
        self,
        dialogues: List[Dict],
        ontology: Ontology,
        max_context_turns: int = 3
    ):
        self.samples = []
        self.ontology = ontology
        self.max_context_turns = max_context_turns
        
        for dialogue in dialogues:
            self._process_dialogue(dialogue)
    
    def _process_dialogue(self, dialogue: Dict):
        """Trích xuất samples từ một dialogue"""
        turns = dialogue.get("turns", [])
        context = []
        
        for turn in turns:
            speaker = turn.get("speaker")
            utterance = turn.get("utterance", "")
            
            if speaker == "USER":
                # Lấy gold state
                gold_states, gold_requests = extract_gold_states({"turns": turns[:turns.index(turn)+1]})
                if gold_states:
                    gold_state = gold_states[-1]
                    gold_req = gold_requests[-1] if gold_requests else []
                    
                    self.samples.append({
                        "dialogue_id": dialogue.get("dialogue_id"),
                        "turn_id": turn.get("turn_id"),
                        "context": context[-self.max_context_turns:],  # giới hạn context
                        "current_utterance": utterance,
                        "gold_state": gold_state,
                        "gold_requested": gold_req
                    })
            
            context.append((speaker, utterance))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def prepare_context(sample: Dict) -> str:
    """Chuẩn bị context string từ sample"""
    context_parts = []
    for speaker, utt in sample["context"]:
        context_parts.append(f"{speaker}: {utt}")
    context_parts.append(f"USER: {sample['current_utterance']}")
    return " ".join(context_parts)


def collate_fn(batch, tokenizer, max_length=512):
    """Collate function để batch samples"""
    contexts = [prepare_context(sample) for sample in batch]
    
    # Tokenize
    encoded = tokenizer(
        contexts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "samples": batch
    }


def get_slot_label(gold_state: Dict, slot: str, slot_values: List[str]) -> int:
    """
    Lấy label index cho closed slot
    0: none, 1: dontcare, 2+: actual values
    """
    if slot not in gold_state:
        return 0  # none
    
    values = gold_state[slot]
    if not values:
        return 0
    
    value = normalize_value(values[0], slot)
    
    if value == "dontcare":
        return 1
    
    # Tìm trong slot_values
    for i, v in enumerate(slot_values):
        if normalize_value(v, slot) == value:
            return i + 2  # offset cho none và dontcare
    
    return 0  # không tìm thấy → none


def get_span_labels(gold_state: Dict, slot: str, tokens: List[str], tokenizer) -> Tuple[int, int]:
    """
    Lấy start/end index cho open slot span
    Return (0, 0) nếu không có value hoặc không tìm thấy trong context
    """
    if slot not in gold_state:
        return 0, 0
    
    values = gold_state[slot]
    if not values:
        return 0, 0
    
    value = normalize_text(values[0])
    
    # Tìm value trong tokens
    # Đơn giản hóa: tìm substring match
    # TODO: Implement proper alignment như SQuAD
    
    return 0, 0  # Placeholder - cần implement token alignment


def get_requested_labels(gold_requested: List[str], all_slots: List[str]) -> torch.Tensor:
    """
    Lấy binary labels cho requested slots
    """
    labels = torch.zeros(len(all_slots))
    for i, slot in enumerate(all_slots):
        if slot in gold_requested:
            labels[i] = 1.0
    return labels


def train_epoch(
    closed_models: Dict[str, SlotClassifier],
    open_models: Dict[str, SpanExtractor],
    requested_model: RequestedSlotsClassifier,
    dataloader: DataLoader,
    optimizers: Dict,
    schedulers: Dict,
    device: torch.device,
    epoch: int,
    ontology: Ontology
):
    """Training một epoch cho tất cả models"""
    
    # Set training mode
    for model in closed_models.values():
        model.train()
    for model in open_models.values():
        model.train()
    requested_model.train()
    
    total_loss = 0
    closed_losses = {slot: 0 for slot in closed_models.keys()}
    open_losses = {slot: 0 for slot in open_models.keys()}
    requested_loss_sum = 0
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        samples = batch["samples"]
        
        batch_size = len(samples)
        
        # ===== CLOSED SLOTS =====
        for slot, model in closed_models.items():
            optimizers[f"closed_{slot}"].zero_grad()
            
            # Forward
            logits = model(input_ids, attention_mask)
            
            # Get labels
            labels = torch.tensor([
                get_slot_label(s["gold_state"], slot, list(ontology.values(slot)))
                for s in samples
            ]).to(device)
            
            # Loss & backward
            loss = criterion_ce(logits, labels)
            loss.backward()
            optimizers[f"closed_{slot}"].step()
            schedulers[f"closed_{slot}"].step()
            
            closed_losses[slot] += loss.item()
        
        # ===== OPEN SLOTS =====
        for slot, model in open_models.items():
            optimizers[f"open_{slot}"].zero_grad()
            
            # Forward
            start_logits, end_logits = model(input_ids, attention_mask)
            
            # Get labels (simplified - actual implementation needs proper alignment)
            start_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
            end_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
            
            # Loss & backward
            loss = criterion_ce(start_logits, start_labels) + criterion_ce(end_logits, end_labels)
            loss.backward()
            optimizers[f"open_{slot}"].step()
            schedulers[f"open_{slot}"].step()
            
            open_losses[slot] += loss.item()
        
        # ===== REQUESTED SLOTS =====
        optimizers["requested"].zero_grad()
        
        # Forward
        logits = requested_model(input_ids, attention_mask)
        
        # Get labels
        labels = torch.stack([
            get_requested_labels(s["gold_requested"], requested_model.slots)
            for s in samples
        ]).to(device)
        
        # Loss & backward
        loss = criterion_bce(logits, labels)
        loss.backward()
        optimizers["requested"].step()
        schedulers["requested"].step()
        
        requested_loss_sum += loss.item()
        
        # Update progress
        avg_loss = (sum(closed_losses.values()) + sum(open_losses.values()) + requested_loss_sum) / (len(dataloader) + 1e-10)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    # Log epoch results
    logging.info(f"Epoch {epoch} completed:")
    logging.info(f"  Closed slots avg loss: {sum(closed_losses.values()) / (len(closed_losses) * len(dataloader)):.4f}")
    logging.info(f"  Open slots avg loss: {sum(open_losses.values()) / (len(open_losses) * len(dataloader) + 1e-10):.4f}")
    logging.info(f"  Requested slots avg loss: {requested_loss_sum / len(dataloader):.4f}")
    
    return total_loss


def predict_turn(
    sample: Dict,
    closed_models: Dict[str, SlotClassifier],
    open_models: Dict[str, SpanExtractor],
    requested_model: RequestedSlotsClassifier,
    tokenizer,
    device: torch.device,
    ontology: Ontology,
    previous_state: Dict
) -> Tuple[Dict, List]:
    """
    Predict state cho một turn
    """
    # Prepare input
    context = prepare_context(sample)
    encoded = tokenizer(
        context,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    predicted_state = {}
    
    # ===== CLOSED SLOTS =====
    for slot, model in closed_models.items():
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            pred_idx = torch.argmax(logits, dim=-1).item()
            
            if pred_idx == 0:  # none - carryover
                if slot in previous_state:
                    predicted_state[slot] = previous_state[slot]
            elif pred_idx == 1:  # dontcare
                predicted_state[slot] = ["dontcare"]
            else:
                value = model.slot_values[pred_idx]
                predicted_state[slot] = [value]
    
    # ===== OPEN SLOTS =====
    for slot, model in open_models.items():
        model.eval()
        with torch.no_grad():
            start_logits, end_logits = model(input_ids, attention_mask)
            start_idx = torch.argmax(start_logits, dim=-1).item()
            end_idx = torch.argmax(end_logits, dim=-1).item()
            
            if start_idx > 0 and end_idx >= start_idx and end_idx < len(input_ids[0]):
                # Extract span
                span_ids = input_ids[0][start_idx:end_idx+1]
                value = tokenizer.decode(span_ids, skip_special_tokens=True)
                predicted_state[slot] = [value]
            elif slot in previous_state:
                # Carryover
                predicted_state[slot] = previous_state[slot]
    
    # ===== REQUESTED SLOTS =====
    requested_model.eval()
    with torch.no_grad():
        logits = requested_model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        predicted_requested = [
            requested_model.slots[i]
            for i in range(len(requested_model.slots))
            if probs[0][i] > 0.5
        ]
    
    return predicted_state, predicted_requested


def evaluate(
    data_dir: Path,
    split: str,
    log_dir: Path,
    checkpoint_dir: Optional[Path] = None,
    device: str = "cpu"
):
    """
    Đánh giá DST v2 trên một split
    """
    split_path = data_dir / split
    log_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dst_v2_{split}_{ts}.log"
    details_file = log_dir / f"dst_v2_{split}_details_{ts}.jsonl"
    summary_file = log_dir / f"dst_v2_{split}_summary_{ts}.json"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    logging.info("="*60)
    logging.info("DST V2 - Per-slot BERT/DistilBERT")
    logging.info("="*60)
    logging.info(f"Split: {split}")
    logging.info(f"Checkpoint: {checkpoint_dir or 'None (random init)'}")
    logging.info(f"Device: {device}")
    
    files = sorted(glob(str(split_path / "*.json")))
    logging.info(f"Tìm thấy {len(files)} file dữ liệu")
    
    # Load dialogues
    all_dialogues = []
    for fp in files:
        with open(fp) as f:
            all_dialogues.extend(json.load(f))
    
    # Tạo dataset
    ontology = Ontology(default_ontology)
    dataset = DSTDataset(all_dialogues, ontology)
    
    logging.info(f"Tổng số samples (USER turns): {len(dataset)}")
    
    # Initialize models
    device_obj = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    closed_models = {}
    open_models = {}
    
    # Closed slots
    for slot in ontology._map.keys():
        values = list(ontology.values(slot))
        model = SlotClassifier("distilbert-base-uncased", values).to(device_obj)
        closed_models[slot] = model
    
    # Open slots (example - customize based on your needs)
    open_slots = ["train-departure", "train-destination", "restaurant-name", "hotel-name"]
    for slot in open_slots:
        model = SpanExtractor("distilbert-base-uncased").to(device_obj)
        open_models[slot] = model
    
    # Requested model
    all_slots = list(ontology._map.keys()) + open_slots
    requested_model = RequestedSlotsClassifier("distilbert-base-uncased", all_slots).to(device_obj)
    
    # Load checkpoint if provided
    if checkpoint_dir and checkpoint_dir.exists():
        logging.info(f"Loading checkpoint from {checkpoint_dir}")
        for slot in closed_models:
            ckpt_path = checkpoint_dir / f"closed_{slot}.pt"
            if ckpt_path.exists():
                closed_models[slot].load_state_dict(torch.load(ckpt_path, map_location=device_obj))
        
        for slot in open_models:
            ckpt_path = checkpoint_dir / f"open_{slot}.pt"
            if ckpt_path.exists():
                open_models[slot].load_state_dict(torch.load(ckpt_path, map_location=device_obj))
        
        req_ckpt = checkpoint_dir / "requested.pt"
        if req_ckpt.exists():
            requested_model.load_state_dict(torch.load(req_ckpt, map_location=device_obj))
    else:
        logging.warning("No checkpoint provided - using random initialization!")
    
    pred_states_all = []
    gold_states_all = []
    pred_requests_all = []
    gold_requests_all = []
    
    # Track state per dialogue
    dialogue_states = {}
    
    with details_file.open("w") as fw:
        for sample in tqdm(dataset, desc="Evaluating"):
            dialogue_id = sample["dialogue_id"]
            
            # Get previous state
            previous_state = dialogue_states.get(dialogue_id, {})
            
            # Predict
            pred_state, pred_req = predict_turn(
                sample, closed_models, open_models, requested_model,
                tokenizer, device_obj, ontology, previous_state
            )
            
            # Update dialogue state
            dialogue_states[dialogue_id] = pred_state
            
            gold_state = sample["gold_state"]
            gold_req = sample["gold_requested"]
            
            pred_states_all.append(pred_state)
            gold_states_all.append(gold_state)
            pred_requests_all.append(pred_req)
            gold_requests_all.append(gold_req)
            
            # Log chi tiết
            detail = {
                "dialogue_id": sample["dialogue_id"],
                "turn_id": sample["turn_id"],
                "utterance": sample["current_utterance"],
                "gold_state": gold_state,
                "pred_state": pred_state,
                "gold_requested": gold_req,
                "pred_requested": pred_req,
                "match": gold_state == pred_state and gold_req == pred_req
            }
            fw.write(json.dumps(detail, ensure_ascii=False, indent=2) + "\n")
    
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
    logging.info(f"Tổng số samples: {len(dataset)}")
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
        "model": "DST_v2_per_slot_bert",
        "split": split,
        "timestamp": ts,
        "checkpoint": str(checkpoint_dir) if checkpoint_dir else None,
        "samples": len(dataset),
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


def save_checkpoint(
    checkpoint_dir: Path,
    closed_models: Dict,
    open_models: Dict,
    requested_model,
    optimizers: Dict,
    epoch: int,
    best_jga: float
):
    """Save model checkpoints"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    for slot, model in closed_models.items():
        torch.save(model.state_dict(), checkpoint_dir / f"closed_{slot}.pt")
    
    for slot, model in open_models.items():
        torch.save(model.state_dict(), checkpoint_dir / f"open_{slot}.pt")
    
    torch.save(requested_model.state_dict(), checkpoint_dir / "requested.pt")
    
    # Save training info
    info = {
        "epoch": epoch,
        "best_jga": best_jga
    }
    with open(checkpoint_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    logging.info(f"Checkpoint saved to {checkpoint_dir}")


def train(
    data_dir: Path,
    log_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = "cpu"
):
    """
    Train DST v2 models
    """
    # Setup logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dst_v2_train_{ts}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    logging.info("="*60)
    logging.info("DST V2 TRAINING")
    logging.info("="*60)
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {lr}")
    logging.info(f"Device: {device}")
    
    # Load training data
    train_path = data_dir / "train"
    files = sorted(glob(str(train_path / "*.json")))
    logging.info(f"Loading {len(files)} training files...")
    
    all_dialogues = []
    for fp in files:
        with open(fp) as f:
            all_dialogues.extend(json.load(f))
    
    ontology = Ontology(default_ontology)
    train_dataset = DSTDataset(all_dialogues, ontology)
    logging.info(f"Training samples: {len(train_dataset)}")
    
    # Load dev data for validation
    dev_path = data_dir / "dev"
    dev_files = sorted(glob(str(dev_path / "*.json")))
    dev_dialogues = []
    for fp in dev_files:
        with open(fp) as f:
            dev_dialogues.extend(json.load(f))
    dev_dataset = DSTDataset(dev_dialogues, ontology)
    logging.info(f"Dev samples: {len(dev_dataset)}")
    
    # Initialize models
    device_obj = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    closed_models = {}
    open_models = {}
    
    # Closed slots
    for slot in ontology._map.keys():
        values = list(ontology.values(slot))
        model = SlotClassifier("distilbert-base-uncased", values).to(device_obj)
        closed_models[slot] = model
        logging.info(f"Initialized closed slot model: {slot} ({len(values)} values)")
    
    # Open slots
    open_slots = ["train-departure", "train-destination", "restaurant-name", "hotel-name"]
    for slot in open_slots:
        model = SpanExtractor("distilbert-base-uncased").to(device_obj)
        open_models[slot] = model
        logging.info(f"Initialized open slot model: {slot}")
    
    # Requested model
    all_slots = list(ontology._map.keys()) + open_slots
    requested_model = RequestedSlotsClassifier("distilbert-base-uncased", all_slots).to(device_obj)
    logging.info(f"Initialized requested model: {len(all_slots)} slots")
    
    # Create dataloader
    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    
    # Setup optimizers and schedulers
    optimizers = {}
    schedulers = {}
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    for slot, model in closed_models.items():
        opt = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        optimizers[f"closed_{slot}"] = opt
        schedulers[f"closed_{slot}"] = scheduler
    
    for slot, model in open_models.items():
        opt = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        optimizers[f"open_{slot}"] = opt
        schedulers[f"open_{slot}"] = scheduler
    
    opt = AdamW(requested_model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    optimizers["requested"] = opt
    schedulers["requested"] = scheduler
    
    logging.info(f"Total training steps: {total_steps}")
    logging.info(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    best_jga = 0.0
    checkpoint_dir = log_dir / f"checkpoints_{ts}"
    
    for epoch in range(1, epochs + 1):
        logging.info("="*60)
        logging.info(f"EPOCH {epoch}/{epochs}")
        logging.info("="*60)
        
        # Train
        train_epoch(
            closed_models,
            open_models,
            requested_model,
            train_loader,
            optimizers,
            schedulers,
            device_obj,
            epoch,
            ontology
        )
        
        # Validate on dev set (quick eval)
        logging.info("Validating on dev set...")
        
        pred_states = []
        gold_states = []
        dialogue_states = {}
        
        for sample in tqdm(dev_dataset, desc="Dev eval"):
            dialogue_id = sample["dialogue_id"]
            previous_state = dialogue_states.get(dialogue_id, {})
            
            pred_state, _ = predict_turn(
                sample, closed_models, open_models, requested_model,
                tokenizer, device_obj, ontology, previous_state
            )
            
            dialogue_states[dialogue_id] = pred_state
            pred_states.append(pred_state)
            gold_states.append(sample["gold_state"])
        
        dev_jga = compute_jga(gold_states, pred_states)
        logging.info(f"Dev JGA: {dev_jga:.4f}")
        
        # Save checkpoint if best
        if dev_jga > best_jga:
            best_jga = dev_jga
            save_checkpoint(
                checkpoint_dir,
                closed_models,
                open_models,
                requested_model,
                optimizers,
                epoch,
                best_jga
            )
            logging.info(f"New best JGA: {best_jga:.4f} - Checkpoint saved!")
    
    logging.info("="*60)
    logging.info("TRAINING COMPLETED")
    logging.info(f"Best dev JGA: {best_jga:.4f}")
    logging.info(f"Checkpoint directory: {checkpoint_dir}")
    logging.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="DST V2: Per-slot BERT/DistilBERT approach"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        required=True,
        help="Training hoặc evaluation"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("datasets/MultiWOZ"),
        help="Thư mục chứa dữ liệu MultiWOZ"
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "train"],
        default="dev",
        help="Split để sử dụng"
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("logs"),
        help="Thư mục lưu log và checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path đến model checkpoint để load"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Số epochs để train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, mps)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(
            args.data_dir,
            args.log_dir,
            args.epochs,
            args.batch_size,
            args.lr,
            args.device
        )
    else:
        evaluate(args.data_dir, args.split, args.log_dir, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
