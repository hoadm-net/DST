"""
Model architectures for DST v2: Per-slot với DistilBERT/BERT

Mỗi slot được model riêng biệt:
- Closed slots: classification
- Open slots: span extraction
- Requested slots: multi-label classification
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SlotClassifier(nn.Module):
    """
    Classifier cho closed slots
    Input: [CLS] context [SEP]
    Output: softmax qua các giá trị + none + dontcare
    """
    def __init__(
        self,
        encoder_name: str,
        slot_values: List[str],
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        
        # none, dontcare + actual values
        self.num_labels = len(slot_values) + 2
        self.slot_values = ["none", "dontcare"] + slot_values
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class SpanExtractor(nn.Module):
    """
    Span extraction cho open slots
    Input: [CLS] context [SEP]
    Output: start/end logits cho từng token
    """
    def __init__(self, encoder_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)
        
        return start_logits, end_logits


class RequestedSlotsClassifier(nn.Module):
    """
    Multi-label classifier cho requested slots
    Input: [CLS] context [SEP]
    Output: sigmoid cho từng slot
    """
    def __init__(
        self,
        encoder_name: str,
        slots: List[str],
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.num_slots = len(slots)
        self.slots = slots
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_slots)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class DSTV2Model:
    """
    Wrapper cho toàn bộ DST model với per-slot approach
    Quản lý nhiều models riêng cho từng loại slot
    """
    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        device: str = "cpu"
    ):
        self.encoder_name = encoder_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # Sẽ khởi tạo khi có ontology
        self.closed_slot_models: Dict[str, SlotClassifier] = {}
        self.open_slot_models: Dict[str, SpanExtractor] = {}
        self.requested_model: Optional[RequestedSlotsClassifier] = None
        
    def init_closed_slot_model(self, slot: str, values: List[str]) -> SlotClassifier:
        """Khởi tạo model cho 1 closed slot"""
        model = SlotClassifier(self.encoder_name, values).to(self.device)
        self.closed_slot_models[slot] = model
        return model
        
    def init_open_slot_model(self, slot: str) -> SpanExtractor:
        """Khởi tạo model cho 1 open slot"""
        model = SpanExtractor(self.encoder_name).to(self.device)
        self.open_slot_models[slot] = model
        return model
        
    def init_requested_model(self, slots: List[str]) -> RequestedSlotsClassifier:
        """Khởi tạo model cho requested slots"""
        model = RequestedSlotsClassifier(self.encoder_name, slots).to(self.device)
        self.requested_model = model
        return model
