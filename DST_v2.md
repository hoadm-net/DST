# DST v2: Per-slot với DistilBERT/BERT

## Tổng quan

DST v2 là một phương pháp learning-based cho Dialogue State Tracking, sử dụng chiến lược per-slot modeling: mỗi slot được xử lý bằng một module phù hợp (classification cho closed slots, span extraction cho open slots), và một module multi-label cho requested slots. Mục tiêu của tài liệu này là mô tả chi tiết mô hình v2, cách chuẩn bị dữ liệu, huấn luyện và suy luận.

### Động lực

- Xử lý paraphrase, cách diễn đạt khác nhau và tham chiếu ngữ cảnh mà các rule-based khó bắt được.
- Học từ dữ liệu để tổng quát hóa và giảm dependency vào các luật thủ công.

### Thiết kế chung

Per-slot modeling cho phép dùng kiến trúc phù hợp cho từng loại slot: closed slots (giá trị nằm trong ontology) dùng classification, open slots (giá trị tự do) dùng span extraction, requested slots dùng multi-label classification.

## Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                    DIALOGUE CONTEXT                         │
│  SYSTEM: Hello, how can I help you?                         │
│  USER: I need a chinese restaurant in the centre            │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   ┌────────────────┐
                   │  Tokenization  │
                   │  [CLS] context │
                   │      [SEP]     │
                   └────────────────┘
                            ↓
              ┌─────────────┴─────────────┐
              │                           │
    ┌─────────▼─────────┐       ┌────────▼─────────┐
    │  DistilBERT/BERT  │       │  DistilBERT/BERT │
    │    Encoder        │       │    Encoder       │
    └─────────┬─────────┘       └────────┬─────────┘
              │                           │
    ┌─────────▼─────────┐       ┌────────▼─────────┐
    │  Closed Slots     │       │   Open Slots     │
    │  (Classification) │       │ (Span Extract)   │
    └─────────┬─────────┘       └────────┬─────────┘
              │                           │
    ┌─────────▼─────────────────────────┬─┘
    │        Requested Slots             │
    │    (Multi-label Classify)          │
    └────────────────┬───────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              PREDICTED BELIEF STATE                         │
│  slot_values: {                                             │
│    "restaurant-food": ["chinese"],      # closed            │
│    "restaurant-area": ["centre"],       # closed            │
│    "train-departure": ["cambridge"]     # open (span)       │
│  }                                                          │
│  requested_slots: ["restaurant-address"] # multi-label     │
└─────────────────────────────────────────────────────────────┘
```

## Chi tiết từng thành phần

### 1. Context Encoding

**Input Preparation**:
```python
# Lấy lịch sử gần nhất (max_context_turns = 3)
context = [
    "SYSTEM: Hello, how can I help?",
    "USER: I need a chinese restaurant",
    "SYSTEM: What area do you prefer?",
    "USER: In the centre"  # current turn
]

# Tokenize
input_text = " ".join(context)
tokens = tokenizer(input_text, max_length=512, truncation=True)
# [CLS] system : hello , how can i help ? user : i need a chinese restaurant ... [SEP]
```

**Shared Encoder**:
- Dùng DistilBERT hoặc BERT-base
- Output: contextualized representations cho mỗi token
- `[CLS]` token: representation cho toàn bộ sequence

### 2. Closed Slots: Classification

**Architecture**:
```
Input: [CLS] context [SEP]
          ↓
    DistilBERT
          ↓
    [CLS] embedding (768-dim)
          ↓
      Dropout(0.1)
          ↓
    Linear(768 → num_labels)
          ↓
       Softmax
          ↓
Output: P(none), P(dontcare), P(value₁), ..., P(valueₙ)
```

**Ví dụ: restaurant-pricerange**
```python
num_labels = 2 + len(values)  # none, dontcare + actual values
           = 2 + 3              # none, dontcare, cheap, moderate, expensive
           = 5

Output logits: [0.1, 0.05, 2.3, -0.5, 0.2]
              ↓ softmax
Probabilities: [0.02, 0.01, 0.92, 0.01, 0.04]
                                ↑
                            argmax → "cheap"
```

**Training**:
- Loss: CrossEntropyLoss
- Label: index của giá trị đúng
  - "none" → 0 (không đề cập slot này)
  - "dontcare" → 1 (người dùng nói "any", "doesn't matter")
  - Actual values → 2, 3, 4, ...

**Carryover Handling**:
- Nếu predict "none" → giữ giá trị từ lượt trước
- Nếu predict value → cập nhật state

### 3. Open Slots: Span Extraction

**Architecture**:
```
Input: [CLS] context [SEP]
          ↓
    DistilBERT
          ↓
Sequence output (batch, seq_len, 768)
          ↓
      Dropout(0.1)
          ↓
  ┌────────────────┬────────────────┐
  │                │                │
Linear(768→1)   Linear(768→1)      │
  │                │                │
Start logits   End logits          │
  └────────────────┴────────────────┘
```

**Ví dụ: train-departure**
```
Tokens:  [CLS] i am leaving from cambridge to norwich [SEP]
Indices:   0   1  2    3     4     5       6   7      8

Start logits: [-2, -1, -2, -3, -4,  3.5, -2, -3,  -5]
End logits:   [-3, -2, -1, -2, -1,  2.8, -3, -2,  -4]
                                      ↑     ↑
                                  start=5, end=5
                                  → "cambridge"
```

**Training**:
- Loss: CrossEntropyLoss cho start + CrossEntropyLoss cho end
- Label: (start_idx, end_idx) của span chính xác
- Nếu value không xuất hiện trong context → start=0, end=0 (no-answer)

**Inference**:
```python
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits)

if start_idx == 0 or end_idx == 0 or start_idx > end_idx:
    predicted_value = "none"  # carryover từ lượt trước
else:
    span = tokens[start_idx:end_idx+1]
    predicted_value = tokenizer.decode(span)
```

### 4. Requested Slots: Multi-label Classification

**Architecture**:
```
Input: [CLS] context [SEP]
          ↓
    DistilBERT
          ↓
    [CLS] embedding (768-dim)
          ↓
      Dropout(0.1)
          ↓
    Linear(768 → num_slots)
          ↓
       Sigmoid
          ↓
Output: P(slot₁), P(slot₂), ..., P(slotₙ)  # mỗi slot độc lập
```

**Ví dụ: Với các possible requested slots**
```python
slots = [
    "restaurant-address",
    "restaurant-phone", 
    "restaurant-postcode",
    "hotel-address",
    ...
]

USER: "I need the address and phone number"

Output: [0.92, 0.88, 0.12, 0.05, ...]  # sigmoid outputs
         ↑     ↑
    threshold=0.5
         
Predicted: ["restaurant-address", "restaurant-phone"]
```

**Training**:
- Loss: BCEWithLogitsLoss (Binary Cross Entropy)
- Label: binary vector [1, 1, 0, 0, ...] cho slots được requested

## Workflow Training

### Bước 1: Chuẩn bị Data

```python
class DSTDataset:
    def __init__(self, dialogues, ontology, max_context_turns=3):
        for dialogue in dialogues:
            context = []
            for turn in dialogue["turns"]:
                if turn["speaker"] == "USER":
                    # Tạo sample
                    sample = {
                        "context": context[-max_context_turns:],
                        "utterance": turn["utterance"],
                        "gold_state": extract_state(turn),
                        "gold_requested": extract_requested(turn)
                    }
                    self.samples.append(sample)
                context.append((turn["speaker"], turn["utterance"]))
```

### Bước 2: Khởi tạo Models

```python
# Cho mỗi closed slot
for slot in closed_slots:
    values = ontology.get_values(slot)
    model = SlotClassifier(
        encoder_name="distilbert-base-uncased",
        slot_values=values
    )
    models[slot] = model

# Cho mỗi open slot
for slot in open_slots:
    model = SpanExtractor(
        encoder_name="distilbert-base-uncased"
    )
    models[slot] = model

# Requested slots
requested_model = RequestedSlotsClassifier(
    encoder_name="distilbert-base-uncased",
    slots=all_slots
)
```

### Bước 3: Training Loop (TODO - đang phát triển)

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Prepare input
        input_ids, attention_mask = tokenize_batch(batch)
        
        # Train closed slots
        for slot, model in closed_slot_models.items():
            logits = model(input_ids, attention_mask)
            labels = get_labels(batch, slot)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Train open slots
        # Train requested model
        # ...
```

### Bước 4: Inference

```python
def predict(context, previous_state):
    tokens = tokenizer(context)
    
    predicted_state = {}
    
    # Closed slots
    for slot, model in closed_slot_models.items():
        logits = model(tokens)
        pred_idx = torch.argmax(logits)
        
        if pred_idx == 0:  # none
            # Carryover
            if slot in previous_state:
                predicted_state[slot] = previous_state[slot]
        else:
            value = model.slot_values[pred_idx]
            predicted_state[slot] = [value]
    
    # Open slots
    for slot, model in open_slot_models.items():
        start_logits, end_logits = model(tokens)
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        
        if start_idx > 0 and end_idx >= start_idx:
            span = decode_span(tokens, start_idx, end_idx)
            predicted_state[slot] = [span]
        elif slot in previous_state:
            predicted_state[slot] = previous_state[slot]
    
    # Requested slots
    logits = requested_model(tokens)
    probs = torch.sigmoid(logits)
    requested = [slot for i, slot in enumerate(all_slots) 
                 if probs[i] > 0.5]
    
    return predicted_state, requested
```

## Ưu điểm

- Học từ dữ liệu: giảm dependency vào rules thủ công và tăng khả năng tổng quát hóa với paraphrase và đa dạng diễn đạt.
- Context-aware: encoder như BERT/DistilBERT cung cấp biểu diễn ngữ cảnh cho từng token/sequence.
- Modular: mỗi loại slot sử dụng module phù hợp (classification/span/multi-label), dễ thử nghiệm và thay thế encoder.

## Hạn chế

- Mỗi slot được mô hình hóa độc lập: không học được mối quan hệ trực tiếp giữa các slot (dependency).
- Tài nguyên: nếu dùng nhiều encoder riêng cho từng slot sẽ tốn bộ nhớ và thời gian inference; cần cân nhắc sharing hoặc lightweight encoder.
- Carryover hiện tại xử lý đơn giản (copy giá trị cũ khi không detect), có thể cải tiến bằng cơ chế học được (update gate).
- Closed slots vẫn phụ thuộc ontology; unseen values cần xử lý bổ sung.

## Cài đặt Dependencies

```bash
# Kích hoạt virtual environment
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Nếu muốn cài thủ công
pip install transformers torch tqdm numpy
```

## Cách sử dụng

### Evaluation Mode
```bash
# Đánh giá trên dev split
python dst_v2.py --mode eval --split dev

# Đánh giá trên test split
python dst_v2.py --mode eval --split test

# Với checkpoint đã train
python dst_v2.py --mode eval --split test --checkpoint logs/best_model.pt
```

### Training Mode (hiện thực)
```bash
# Train với default settings
python dst_v2.py --mode train --epochs 3 --batch_size 8

# Custom hyperparameters
python dst_v2.py --mode train --epochs 5 --batch_size 16 --lr 5e-5
```

## Cấu trúc Code (tóm tắt)

- `mint/dst/model.py`: định nghĩa các lớp mạng (SlotClassifier, SpanExtractor, RequestedSlotsClassifier, DSTV2Model).
- `dst_v2.py`: dataset, training loop, evaluation, entrypoint.

## Output và Logging

- Console logs: `logs/dst_v2_{mode}_{YYYYMMDD_HHMMSS}.log`
- Turn-level details: `logs/dst_v2_{split}_details_{ts}.jsonl` (mỗi dòng một turn với gold/pred)
- Summary metrics: `logs/dst_v2_{split}_summary_{ts}.json` (JGA, Slot F1, Requested F1)

## Roadmap Phát triển (gói cho v2)

### Phase A: Skeleton (Hoàn thành)
- Model architectures, dataset, evaluation pipeline, logging.

### Phase B: Training
- Implement đầy đủ training loop (optimizer, scheduler, mixed precision tùy chọn).
- Checkpoint save/load, resume, early stopping theo dev JGA.

### Phase C: Experiments
- Train trên full train split, tune hyperparameters (lr, batch_size, epochs).
- Thử share encoder vs separate encoders.
- Ablation: context window size, label smoothing, loss weighting.

### Phase D: Analysis
- Error analysis per slot, confusion matrix cho closed slots, span extraction metrics.

## Thảo luận (vấn đề và lựa chọn thiết kế)

1. Share encoder vs individual encoders:
    - Share encoder giảm memory & có thể học chung biểu diễn; individual encoder cho mỗi slot có thể tối ưu hóa tốt hơn cho task cụ thể.
2. Training strategy:
    - Joint training (multi-task) hay per-slot separate training? Joint có thể tận dụng shared signal, nhưng cần điều chỉnh loss weighting.
3. Carryover mechanism:
    - Simple copy vs learned update gate (có thể dùng small classifier/regressor hoặc gating network).
4. Handling unseen values for closed slots:
    - Use fallback span-extraction or hybrid approach (classification + open candidate extraction).

## Kết luận

Tài liệu này tập trung mô tả kiến trúc và workflow của DST v2 (per-slot BERT-based). Mục tiêu là có một baseline neural rõ ràng, reproducible để tiến hành huấn luyện, thử nghiệm và tối ưu hóa trước khi xem xét các mở rộng kiến trúc.
