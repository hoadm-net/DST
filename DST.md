# Dialogue State Tracking (DST)

Tài liệu này giải thích chi tiết bài toán Dialogue State Tracking trên MultiWOZ (nhấn mạnh phiên bản v2.2) và minh họa bằng một ví dụ thực tế từ tập `dev/dialogues_001.json`.

## Mục tiêu DST
DST theo dõi và cập nhật đầy đủ trạng thái hội thoại (belief state) sau mỗi lượt của người dùng (USER). Trạng thái là tập ràng buộc slot–value đang “đúng” ở thời điểm đó, phân theo miền (domain), kèm các slot mà người dùng đang yêu cầu hệ thống cung cấp (requested slots).

## Biểu diễn trạng thái (v2.2)
- Slot được chuẩn hóa và “phẳng hóa” theo khóa `domain-slot` (ví dụ: `restaurant-food`, `train-leaveat`).
- `slot_values`: ánh xạ `"domain-slot" -> [list giá trị]` (một số slot có thể có nhiều giá trị).
- `requested_slots`: danh sách slot mà người dùng đang hỏi (ví dụ: `restaurant-address`).
- Trạng thái thường được gắn ở lượt USER; ở v2.2, nhãn được làm sạch và đồng nhất để đánh giá đáng tin cậy.

## Đầu vào và đầu ra của mô hình
- Đầu vào (tại lượt USER t):
  - Lịch sử hội thoại đến thời điểm t (các phát ngôn USER/SYSTEM gần đây; có thể kèm dialogue acts của hệ thống ở lượt t-1).
  - Tùy mô hình: trạng thái ở lượt t-1 để cập nhật gia tăng (carryover + state_update).
- Đầu ra: 
  - `slot_values` đầy đủ ở lượt t (toàn bộ trạng thái hiện thời, không chỉ phần thay đổi).
  - `requested_slots` ở lượt t (nếu có).

## Cách xây dựng mẫu huấn luyện
- Mỗi lượt USER là một mẫu huấn luyện/đánh giá.
- Nhãn mục tiêu là `state` tại lượt đó: `slot_values` và `requested_slots`.
- Có thể suy ra `state_update = state_t − state_{t−1}` để huấn luyện cơ chế cập nhật gia tăng.

## Chỉ số đánh giá
- Joint Goal Accuracy (JGA): dự đoán đúng nếu toàn bộ `slot_values` trùng khớp với ground truth sau chuẩn hóa; sai khác 1 slot/value bất kỳ → 0 cho lượt đó. Điểm là trung bình trên các lượt USER.
- Slot F1 (micro): xem mỗi cặp `domain-slot=value` là một nhãn; tính TP/FP/FN gộp rồi ra Precision/Recall/F1.
- Requested Slots F1: tương tự cho phát hiện `requested_slots`.
- Chuẩn hóa giá trị (v2.2): lowercase, bỏ khoảng trắng thừa, chuẩn thời gian/ngày, ánh xạ đồng nghĩa; slot đóng so khớp theo giá trị canonical.

## Các cách thiết kế mô hình (tóm tắt)
- Heuristic + ontology (không huấn luyện): khớp chuỗi/regex + carryover; dễ triển khai nhưng độ chính xác hạn chế.
- Per-slot với DistilBERT/BERT nhỏ:
  - Slot đóng: phân loại trên tập giá trị ∪ {none, dontcare}.
  - Slot mở: dự đoán span trong ngữ cảnh (hoặc sinh chuỗi) + cơ chế no-answer và carryover.
  - Requested slots: multi-label classification.
- Kiến trúc cổ điển/seq2seq (TRADE, TripPy, SimpleTOD, T5...): mạnh nhưng phức tạp hơn để bắt đầu.

## Ví dụ minh họa từ MultiWOZ (dev/dialogues_001.json)
Đoạn hội thoại rút gọn dưới đây minh họa cách trạng thái được cập nhật qua các lượt USER. Các slot được viết theo chuẩn `domain-slot` của v2.2.

- Thông tin đối thoại: `dialogue_id = PMUL0698.json`, domains liên quan: restaurant, train.

1) Lượt USER t=0
- Utterance: "I'm looking for a local place to dine in the centre that serves chinese food."
- Ground truth state:
  - slot_values:
    - restaurant-area = ["centre"]
    - restaurant-food = ["chinese"]
  - requested_slots: []
- Mô hình cần dự đoán: hai ràng buộc ở domain restaurant như trên.

2) Lượt SYSTEM t=1
- Hệ thống hỏi về giá (price range). Không đánh giá DST ở lượt SYSTEM.

3) Lượt USER t=2
- Utterance: "I need the address, postcode and the price range."
- Ground truth state:
  - slot_values (giữ nguyên ràng buộc tìm nhà hàng):
    - restaurant-area = ["centre"], restaurant-food = ["chinese"]
  - requested_slots:
    - restaurant-address, restaurant-postcode, restaurant-pricerange
- Mô hình cần dự đoán: vẫn giữ đúng ràng buộc và bổ sung requested_slots như trên.

4) Lượt SYSTEM t=3
- Hệ thống cung cấp một gợi ý nhà hàng cụ thể kèm địa chỉ, postcode, giá. Không đánh giá DST ở lượt SYSTEM.

5) Lượt USER t=4
- Utterance: "I also need a train. The train should leave after 16:15 and should leave on sunday."
- Ground truth state:
  - restaurant (giữ nguyên): area = ["centre"], food = ["chinese"]
  - train: train-leaveat = ["16:15"], train-day = ["sunday"]
  - requested_slots: []
- Mô hình cần dự đoán: hợp nhất trạng thái nhiều miền (restaurant + train) với các giá trị mới được nêu ở lượt này.

6) Lượt SYSTEM t=5
- Hệ thống hỏi thêm về điểm đi/điểm đến cho tàu.

7) Lượt USER t=6
- Utterance: "I am leaving from Cambridge and going to Norwich."
- Ground truth state (bổ sung thông tin tàu):
  - restaurant: area = ["centre"], food = ["chinese"]
  - train:
    - train-leaveat = ["16:15"], train-day = ["sunday"],
    - train-departure = ["cambridge"], train-destination = ["norwich"]
  - requested_slots: []
- Mô hình cần dự đoán: thêm `train-departure`, `train-destination` đúng và bảo toàn các giá trị trước đó.

8) Lượt SYSTEM t=7
- Hệ thống đề xuất chuyến TR1840 rời lúc 16:36.

9) Lượt USER t=8
- Utterance: "book for 5 people and get me the reference number"
- Ground truth state (yêu cầu đặt chỗ + mã tham chiếu):
  - restaurant: area = ["centre"], food = ["chinese"]
  - train:
    - train-leaveat = ["16:15"], train-day = ["sunday"],
    - train-departure = ["cambridge"], train-destination = ["norwich"],
    - train-bookpeople = ["5"]
  - requested_slots:
    - train-ref
- Mô hình cần dự đoán: cập nhật thêm `train-bookpeople = 5` và nhận diện requested slot `train-ref`.

10) Lượt SYSTEM t=9
- Hệ thống trả về mã đặt chỗ (reference). Không đánh giá DST ở lượt SYSTEM.

11) Lượt USER t=10
- Utterance: "No, this is all I will need. Thank you."
- Ground truth state: giữ nguyên các slot đã xác lập (không thêm yêu cầu mới).

Ghi chú:
- Các giá trị trong ví dụ đã ở dạng chuẩn hóa (lowercase, chuẩn thời gian), phù hợp quy ước v2.2.
- Với slot có nhiều giá trị (list), cần đảm bảo dự đoán trùng tập giá trị sau chuẩn hóa, thứ tự không quan trọng.

## Hợp đồng tối thiểu (implementer checklist)
- Input: lịch sử hội thoại (cửa sổ ngắn), lượt USER hiện tại, và (tùy chọn) state lượt trước.
- Output: state đầy đủ ở lượt hiện tại: `slot_values` + `requested_slots`.
- Sai số cần lưu ý: phân biệt `none` vs `dontcare`; chuẩn hóa thời gian/ngày; ánh xạ số từ chữ sang số; nhiều miền cùng lúc; slot đa giá trị.
- Tiêu chí thành công: JGA, Slot F1, Requested F1 cải thiện rõ rệt so với heuristic baseline.

## Thực hành khuyến nghị
- Dùng cùng pipeline chuẩn hóa như dữ liệu v2.2 (lowercase + strip + chuẩn thời gian + đồng nghĩa).
- Tận dụng `state_update` (nếu có) để học cập nhật gia tăng và cải thiện ổn định.
- Log riêng phần requested slots để dễ phân tích lỗi.
