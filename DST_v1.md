# DST v1: Rule-based Baseline với Ontology

## Tổng quan

DST v1 là phương án baseline đơn giản sử dụng **heuristics + ontology** để theo dõi trạng thái hội thoại (belief state tracking). Đây là phương pháp không cần huấn luyện (zero-shot), dựa hoàn toàn vào:
- **Ontology**: Danh sách các giá trị hợp lệ cho closed slots
- **Regex patterns**: Trích xuất thông tin có cấu trúc (thời gian, số lượng, địa điểm)
- **Keyword matching**: Phát hiện requested slots
- **Carryover mechanism**: Duy trì state qua các lượt

## Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                     USER UTTERANCE                          │
│         "I need a chinese restaurant in the centre"         │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌──────────────┐
                    │ Normalize    │
                    │ - Lowercase  │
                    │ - Strip      │
                    │ - Synonyms   │
                    └──────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Extract Closed│  │Extract Open  │  │Extract       │
│Slots         │  │Slots         │  │Requested     │
│(Ontology)    │  │(Regex)       │  │(Keywords)    │
└──────────────┘  └──────────────┘  └──────────────┘
        ↓                   ↓                   ↓
        └───────────────────┼───────────────────┘
                            ↓
                  ┌──────────────────┐
                  │ Carryover State  │
                  │ (từ lượt trước)  │
                  └──────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PREDICTED BELIEF STATE                         │
│  slot_values: {"restaurant-food": ["chinese"],              │
│                "restaurant-area": ["centre"]}               │
│  requested_slots: []                                        │
└─────────────────────────────────────────────────────────────┘
```

## Các thành phần chính

### 1. Ontology

Định nghĩa trong `mint/dst/ontology.py` - bao gồm các closed slots với tập giá trị hữu hạn:

```python
default_ontology = {
    # Restaurant
    "restaurant-pricerange": {"cheap", "moderate", "expensive", "dontcare"},
    "restaurant-area": {"centre", "north", "south", "east", "west", "dontcare"},
    
    # Hotel
    "hotel-pricerange": {"cheap", "moderate", "expensive", "dontcare"},
    "hotel-area": {"centre", "north", "south", "east", "west", "dontcare"},
    "hotel-parking": {"yes", "no", "dontcare"},
    "hotel-internet": {"yes", "no", "dontcare"},
    "hotel-stars": {"0", "1", "2", "3", "4", "5", "dontcare"},
    
    # Train
    "train-day": {"monday", "tuesday", ..., "sunday", "dontcare"},
}
```

### 2. Normalization (Chuẩn hóa)

Trong `mint/dst/normalize.py`:

```python
def normalize_text(s: str) -> str:
    - Lowercase
    - Strip whitespace
    - Collapse multiple spaces
    - Apply synonyms: "center" → "centre", "don't care" → "dontcare"
    
def normalize_value(slot: str, value: str) -> str:
    - Normalize text
    - Special handling cho time slots: "16:15" → "16:15"
    - Special handling cho day slots
    - Boolean normalization: "yes"/"true" → "yes"
```

### 3. State Tracking

Class `DSTV1` duy trì:
- `self.state`: Dict[slot, List[value]] - trạng thái hiện tại
- `self.requested`: List[slot] - các slot đang được yêu cầu
- `self.active_domains`: List[domain] - lịch sử domain đã đề cập

## Giải thuật chi tiết

### Bước 1: Khởi tạo
```python
dst = DSTV1(ontology)
# state = {}
# requested = []
# active_domains = []
```

### Bước 2: Xử lý mỗi lượt USER

Cho mỗi user utterance:

#### 2.1. Normalize
```python
text = normalize_text(utterance)
# "I Need A Chinese Restaurant" → "i need a chinese restaurant"
```

#### 2.2. Extract Closed Slots
```python
def _extract_closed_slots(text):
    for slot, values in ontology:
        for value in values:
            if f" {value} " in f" {text} ":
                _add_value(slot, value)
```

**Ví dụ**:
- Text: `"i need a chinese restaurant"`
- Ontology có `restaurant-food: {chinese, indian, ...}`
- Tìm thấy `" chinese "` → Add `restaurant-food = ["chinese"]`

#### 2.3. Extract Open Slots (Regex)

**Thời gian** (train-leaveat):
```python
pattern = r"\b(\d{1,2})[:\.](\d{2})\b"
# "leave after 16:15" → match "16:15" → "16:15"
```

**Số người** (train-bookpeople):
```python
pattern = r"\b(?:for\s+)?(\d{1,2})\s+(?:people|persons|adults)\b"
# "for 5 people" → match "5"
```

**Điểm đi/đến**:
```python
from_pattern = r"\bfrom\s+([a-zA-Z ]{2,})\b"
to_pattern = r"\bto\s+([a-zA-Z ]{2,})\b"
# "from cambridge to norwich" → departure="cambridge", destination="norwich"
```

#### 2.4. Extract Requested Slots

```python
req_keywords = {
    "address": "address",
    "postcode": "postcode", 
    "phone": "phone",
    "price": "pricerange",
    "reference": "ref"
}

def _extract_requested(text):
    domain = _guess_domain(text) or "restaurant"
    for keyword, slotname in req_keywords:
        if keyword in text:
            _set_requested(f"{domain}-{slotname}")
```

**Domain guessing**:
1. Nếu có `active_domains` → dùng domain gần nhất
2. Nếu không, dò từ khóa:
   - "train", "depart", "leave" → "train"
   - "restaurant", "food", "dine" → "restaurant"
   - "hotel" → "hotel"

#### 2.5. Carryover State

State được **duy trì** qua các lượt (không reset):
```python
# Turn 1: restaurant-area = ["centre"]
# Turn 2: restaurant-food = ["chinese"]
# → State sau turn 2: {
#     "restaurant-area": ["centre"],    # carryover
#     "restaurant-food": ["chinese"]    # mới
#   }
```

### Bước 3: Output
```python
return (
    {slot: values[:] for slot, values in self.state.items()},  # copy
    self.requested[:]  # copy
)
```

## Case Study: Đặt nhà hàng và tàu

### Dialogue Context
```
SYSTEM: Hello, how can I help you today?
```

### Turn 1
**USER**: "I'm looking for a chinese restaurant in the centre"

**Xử lý**:
1. Normalize: `"i'm looking for a chinese restaurant in the centre"`
2. Extract closed slots:
   - Tìm `" chinese "` → `restaurant-food = ["chinese"]`
   - Tìm `" centre "` → `restaurant-area = ["centre"]`
3. Extract open slots: (không có)
4. Extract requested: (không có)
5. Update active_domains: `["restaurant"]`

**Output**:
```json
{
  "slot_values": {
    "restaurant-food": ["chinese"],
    "restaurant-area": ["centre"]
  },
  "requested_slots": []
}
```

---

### Turn 2
**SYSTEM**: "I have restaurants matching your criteria. Do you have a price preference?"

**USER**: "I need the address and phone number"

**Xử lý**:
1. Normalize: `"i need the address and phone number"`
2. Extract closed slots: (không có)
3. Extract open slots: (không có)
4. Extract requested:
   - Domain = "restaurant" (từ active_domains)
   - Tìm `"address"` → `restaurant-address`
   - Tìm `"phone"` → `restaurant-phone`
5. State không đổi (carryover)

**Output**:
```json
{
  "slot_values": {
    "restaurant-food": ["chinese"],      # carryover
    "restaurant-area": ["centre"]        # carryover
  },
  "requested_slots": ["restaurant-address", "restaurant-phone"]
}
```

---

### Turn 3
**SYSTEM**: "Charlie Chan is located at Regent Street City Centre. Phone is 01223..."

**USER**: "I also need a train. The train should leave after 16:15 on sunday"

**Xử lý**:
1. Normalize: `"i also need a train. the train should leave after 16:15 on sunday"`
2. Extract closed slots:
   - Tìm `" sunday "` → `train-day = ["sunday"]`
3. Extract open slots:
   - Regex time: `"16:15"` → `train-leaveat = ["16:15"]`
4. Extract requested: (không có)
5. Update active_domains: `["restaurant", "train"]`

**Output**:
```json
{
  "slot_values": {
    "restaurant-food": ["chinese"],      # carryover
    "restaurant-area": ["centre"],       # carryover
    "train-day": ["sunday"],             # new
    "train-leaveat": ["16:15"]           # new
  },
  "requested_slots": []
}
```

---

### Turn 4
**SYSTEM**: "Where are you departing from and going to?"

**USER**: "I am leaving from Cambridge and going to Norwich"

**Xử lý**:
1. Normalize: `"i am leaving from cambridge and going to norwich"`
2. Extract closed slots: (không có)
3. Extract open slots:
   - Regex from: `"from cambridge"` → `train-departure = ["cambridge"]`
   - Regex to: `"to norwich"` → `train-destination = ["norwich"]`
4. Extract requested: (không có)

**Output**:
```json
{
  "slot_values": {
    "restaurant-food": ["chinese"],           # carryover
    "restaurant-area": ["centre"],            # carryover
    "train-day": ["sunday"],                  # carryover
    "train-leaveat": ["16:15"],              # carryover
    "train-departure": ["cambridge"],         # new
    "train-destination": ["norwich"]          # new
  },
  "requested_slots": []
}
```

---

### Turn 5
**SYSTEM**: "I have train TR1840 leaving at 16:36. Would you like me to book it?"

**USER**: "book for 5 people and get me the reference number"

**Xử lý**:
1. Normalize: `"book for 5 people and get me the reference number"`
2. Extract closed slots: (không có)
3. Extract open slots:
   - Regex people: `"for 5 people"` → `train-bookpeople = ["5"]`
4. Extract requested:
   - Domain = "train" (từ active_domains[-1])
   - Tìm `"reference"` → `train-ref`

**Output**:
```json
{
  "slot_values": {
    "restaurant-food": ["chinese"],           # carryover
    "restaurant-area": ["centre"],            # carryover
    "train-day": ["sunday"],                  # carryover
    "train-leaveat": ["16:15"],              # carryover
    "train-departure": ["cambridge"],         # carryover
    "train-destination": ["norwich"],         # carryover
    "train-bookpeople": ["5"]                 # new
  },
  "requested_slots": ["train-ref"]
}
```

## Ưu điểm

✅ **Đơn giản, dễ hiểu**: Code rõ ràng, dễ debug
✅ **Không cần training**: Chạy ngay không cần dữ liệu huấn luyện
✅ **Nhanh**: Inference rất nhanh (không có model)
✅ **Interpretable**: Biết chính xác tại sao model dự đoán như vậy
✅ **Baseline tốt**: Cung cấp lower bound để so sánh

## Hạn chế

❌ **Phụ thuộc ontology**: Chỉ hoạt động với slot/value đã định nghĩa trước
❌ **Regex cứng nhắc**: Không xử lý được biến thể phức tạp
   - "a quarter to twelve" ❌
   - "around 11:45" ❌ (match được nhưng "around" không bỏ)
❌ **Domain guessing đơn giản**: Dễ sai khi multi-domain
❌ **Không học từ data**: Không cải thiện theo thời gian
❌ **Requested slots hạn chế**: Chỉ dựa vào từ khóa trực tiếp
   - "where is it?" ❌ (không có keyword "address")
   - "how much does it cost?" ❌ (không match "price")

## Kết quả thực nghiệm

Trên **test split** (1000 dialogues, 7372 USER turns):

| Metric | Score |
|--------|-------|
| **JGA** (Joint Goal Accuracy) | **1.41%** |
| **Slot F1** | **22.83%** |
| - Precision | 23.66% |
| - Recall | 22.06% |
| **Requested F1** | **5.15%** |
| - Precision | 3.35% |
| - Recall | 11.15% |

**Nhận xét**:
- JGA rất thấp (1.41%) vì phải đúng **toàn bộ** state
- Slot F1 khá hơn (22.83%) vì được tính theo từng cặp slot-value
- Requested F1 rất thấp (5.15%) do keyword matching đơn giản

## Cải tiến có thể

1. **Mở rộng ontology**: Thêm nhiều giá trị closed slots hơn
2. **Better regex**: Xử lý nhiều format thời gian/ngày hơn
3. **Synonym expansion**: "center" = "centre", "2" = "two"
4. **Context-aware requested**: Dùng system turn trước để đoán requested
5. **Coreference resolution**: "it" → nhà hàng/khách sạn vừa đề cập

## Cách chạy

```bash
# Evaluate trên test split
python dst_v1.py --split test

# Evaluate trên dev split
python dst_v1.py --split dev

# Custom log directory
python dst_v1.py --split test --log_dir my_logs
```

## Output files

- `logs/dst_v1_test_YYYYMMDD_HHMMSS.log`: Log đầy đủ
- `logs/dst_v1_test_details_YYYYMMDD_HHMMSS.jsonl`: Chi tiết từng turn
- `logs/dst_v1_test_summary_YYYYMMDD_HHMMSS.json`: Metrics tổng hợp

## Kết luận

DST v1 cung cấp một baseline đơn giản nhưng hiệu quả để:
- **Hiểu bản chất bài toán DST**: Input/output, carryover, normalization
- **Đánh giá độ khó**: JGA 1.41% cho thấy DST không dễ
- **Baseline so sánh**: Cho các phương pháp phức tạp hơn (v2: BERT, v3: GNN)

Phương pháp này phù hợp cho:
- Prototype nhanh
- Miền hẹp với ontology nhỏ
- Khi không có dữ liệu training
- Baseline để đo lường tiến bộ của learning-based methods
