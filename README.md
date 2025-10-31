# Dialogue State Tracking (DST)

Nghiên cứu bài toán Dialogue State Tracking (DST) trên dataset MultiWOZ v2.2.

## Mục tiêu

### 1. Tìm hiểu bài toán DST
Hiểu rõ bài toán theo dõi trạng thái hội thoại (Dialogue State Tracking) trong hệ thống hội thoại nhiệm vụ (task-oriented dialogue):

- **Input/Output**: Nắm vững đầu vào (lịch sử hội thoại) và đầu ra (belief state: slot_values + requested_slots)
- **Đánh giá**: Các độ đo Joint Goal Accuracy (JGA), Slot F1, Requested Slots F1
- **Chuẩn hóa**: Normalization, ontology, carryover state
- **Baseline**: Xây dựng phương pháp rule-based đơn giản để hiểu bản chất bài toán

### 2. Tìm hiểu và ứng dụng GNN cho bài toán DST
Nghiên cứu cách áp dụng Graph Neural Network (GNN) vào DST:

- **Graph representation**: Biểu diễn hội thoại dưới dạng đồ thị
- **GNN architectures**: Khảo sát các kiến trúc GNN phù hợp
- **Implementation**: Hiện thực và đánh giá phương pháp dựa trên GNN
- **Comparison**: So sánh với baseline và các phương pháp hiện đại

## Bài toán DST

**Định nghĩa**: Tại mỗi lượt người dùng (USER turn) trong hội thoại, dự đoán belief state đầy đủ bao gồm:
- `slot_values`: Dict[domain-slot, List[value]] - các ràng buộc hiện tại
- `requested_slots`: List[domain-slot] - các slot người dùng đang yêu cầu

**Ví dụ**:
```
USER: "I'm looking for a chinese restaurant in the centre"
→ State: {
    "restaurant-food": ["chinese"],
    "restaurant-area": ["centre"]
  }

USER: "I need the address and phone number"
→ State không đổi, requested_slots: ["restaurant-address", "restaurant-phone"]
```

**Đánh giá**:
- **JGA (Joint Goal Accuracy)**: % lượt dự đoán đúng hoàn toàn toàn bộ state
- **Slot F1**: Precision/Recall/F1 trên từng cặp (slot, value)
- **Requested F1**: F1 cho việc phát hiện requested slots

## Dataset: MultiWOZ v2.2

- **Quy mô**: ~10,000 hội thoại, 7 domains (restaurant, hotel, train, taxi, attraction, police, hospital)
- **Cải tiến v2.2**: Chuẩn hóa belief state (phẳng hóa domain-slot), sửa lỗi nhãn, đồng bộ giá trị
- **Split**: train/dev/test với các file dialogues_*.json

Chi tiết về dataset và cấu trúc dữ liệu: xem [MultiWOZ.md](MultiWOZ.md) và [DST.md](DST.md)

## Roadmap phương án

### Phương án 1: Heuristic + Ontology (hoàn thành)
- Rule-based baseline với ontology nhỏ
- Mục đích: Hiểu bài toán, làm baseline so sánh
- Script: `dst_v1.py`

### Phương án 2: Per-slot với DistilBERT/BERT (tiếp theo)
- Phân loại cho closed slots
- Span extraction cho open slots
- Multi-label cho requested slots

### Phương án 3: GNN-based approach (mục tiêu chính)
- Biểu diễn hội thoại dưới dạng đồ thị
- Áp dụng GNN để học representation
- Khai thác cấu trúc và quan hệ trong dialogue
