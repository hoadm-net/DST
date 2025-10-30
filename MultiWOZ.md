## MultiWOZ: Bản gốc vs. MultiWOZ v2.2

Tài liệu này tóm tắt MultiWOZ (bản gốc) và các cập nhật quan trọng đến phiên bản v2.2, đồng thời giải thích bài toán DST, đầu vào/đầu ra và cách đánh giá.

### 1) Tổng quan MultiWOZ (bản gốc / 2.0)
- Bộ dữ liệu hội thoại nhiệm vụ (task-oriented), đa miền, đa lượt, người–người (Wizard-of-Oz).
- Quy mô: ~10,438 hội thoại, >115k lượt thoại, 7 miền chính: Attraction, Hotel, Restaurant, Taxi, Train, Police, Hospital.
- Chú giải gồm: phát ngôn người dùng/hệ thống, trạng thái hội thoại (belief state: các ràng buộc slot–value), hành động hội thoại (dialogue acts), và mục tiêu tác vụ.
- Cấu trúc belief state ban đầu (2.0): mỗi miền có hai nhánh con là semi (ràng buộc) và book (đặt chỗ), với tên slot chưa nhất quán, nhiều sai sót/không đồng nhất về chính tả, định dạng giá trị (đặc biệt thời gian, địa chỉ, viết hoa/thường).

Lưu ý chất lượng nhãn:
- 2.0 chứa nhiều lỗi gán nhãn và không nhất quán giữa các lượt, ảnh hưởng mạnh đến đánh giá DST (Joint Goal Accuracy).

### 2) Các bản cập nhật 2.1 và 2.2 (tóm tắt khác biệt)
- 2.1: sửa một lượng lớn lỗi của 2.0 (đặc biệt belief state), nhưng vẫn còn nhiễu.
- 2.2: làm sạch sâu và chuẩn hóa toàn diện:
	- “Phẳng hóa” belief state về dạng khóa domain-slot thống nhất (không còn tách semi/book như 2.0/2.1).
	- Chuẩn hóa giá trị (lowercase nhất quán, chuẩn thời gian/ngày, xử lý đồng nghĩa, bỏ khoảng trắng thừa, v.v.).
	- Đồng bộ span thông tin (nếu cung cấp) giữa giá trị và vị trí xuất hiện trong phát ngôn.
	- Làm rõ requested slots và giảm lỗi ở dialogue acts.
	- Mục tiêu: đánh giá DST tin cậy hơn (JGA phản ánh đúng tiến bộ mô hình, ít “rơi điểm” do nhiễu nhãn).

### 3) Cấu trúc thư mục trong workspace này
Thư mục bạn đang mở: `datasets/MultiWOZ/`
- `train/`, `dev/`, `test/`: chứa các file `dialogues_XXX.json` (mỗi file là một mảng các hội thoại).
- `dialog_acts.json`: ánh xạ/nhãn hành động hội thoại (tùy bản phân phối đi kèm).
- `schema.json`: tệp schema (không phải thành phần chuẩn của MultiWOZ gốc, có thể dùng cho chuyển đổi/tiện ích nội bộ).
- `convert_to_multiwoz_format.py`: script hỗ trợ chuyển/chuẩn hóa dữ liệu về định dạng MultiWOZ.
- `README.md`, `requirements.txt`: mô tả sử dụng và phụ thuộc.

Lưu ý: MultiWOZ bản công bố thường cung cấp ontology (danh sách miền/slot/giá trị đóng). Repo này không hiển thị `ontology.json`, nhưng các tên miền và slot vẫn nhất quán theo quy ước v2.2.

### 4) Cấu trúc dữ liệu hội thoại (đặc biệt trong v2.2)
Mỗi file `dialogues_XXX.json` chứa một mảng hội thoại. Một hội thoại thường có:
- `dialogue_id`: mã hội thoại.
- `domains`: các miền xuất hiện (vd: hotel, restaurant, train, attraction, taxi, hospital, police).
- `turns`: danh sách lượt thoại theo thời gian. Mỗi lượt có thể gồm:
	- `speaker`: "USER" hoặc "SYSTEM".
	- `utterance`: câu nói dạng chuỗi.
	- `turn_idx`: số thứ tự lượt.
	- `dialogue_acts`: hành động hội thoại có cấu trúc (inform, request, confirm, offer, ...), kèm cặp [slot, value].
	- `span_info` (nếu có): vị trí chuỗi giá trị trong phát ngôn.
	- `state` (đáng tin cậy nhất ở lượt USER trong v2.2):
		- `slot_values`: ánh xạ `"domain-slot" -> [list giá trị]` (ví dụ: `"hotel-pricerange": ["cheap"]`, `"train-people": ["2"]`).
		- `requested_slots`: danh sách slot đang được hỏi đến (ví dụ: `"restaurant-address"`).
	- (tùy dữ liệu) `state_update`: phần thay đổi so với lượt trước (tiện cho huấn luyện cập nhật gia tăng).

Khác biệt then chốt v2.2:
- Slot được đặt tên thống nhất theo `domain-slot` (không còn semi/book), giúp đơn giản hóa trích xuất/so khớp.
- Giá trị được chuẩn hóa giúp so sánh chính xác hơn khi đánh giá.

### 5) Bài toán DST (Dialogue State Tracking)
Mục tiêu: theo dõi và cập nhật đầy đủ trạng thái hội thoại (belief state) tại mỗi lượt người dùng.

- Đầu vào (tại một lượt USER):
	- Lịch sử hội thoại (USER và SYSTEM) đến thời điểm hiện tại.
	- Tùy mô hình: trạng thái ở lượt trước và/hoặc hành động hệ thống lượt trước (giúp hiểu xác nhận/hỏi thêm của hệ thống).

- Đầu ra cần dự đoán:
	- `slot_values`: toàn bộ ràng buộc hiện thời dưới dạng cặp `domain-slot = value(s)`.
	- `requested_slots`: các slot người dùng đang yêu cầu hệ thống cung cấp.

- Cách xây dựng mẫu huấn luyện: mỗi lượt USER là một mẫu, với nhãn là `state` hiện tại (có thể suy ra `state_update` = chênh lệch so với lượt trước để huấn luyện cập nhật gia tăng).

### 6) Đánh giá DST trên MultiWOZ v2.2
Chỉ số phổ biến:
- Joint Goal Accuracy (JGA):
	- Một lượt USER được tính đúng nếu toàn bộ `slot_values` dự đoán khớp hoàn toàn với ground truth (sau chuẩn hóa). Sai bất kỳ slot/value nào đều tính 0 cho lượt đó.
	- JGA là trung bình trên các lượt USER của tập dev/test.
- Slot F1 (micro):
	- Xem mỗi cặp `domain-slot=value` là một nhãn; tính TP/FP/FN trên toàn bộ lượt → Precision/Recall/F1.
	- Khoan dung hơn JGA, phản ánh mô hình có thể đúng phần lớn slot.
- Requested Slots F1: tương tự Slot F1 nhưng áp dụng cho phát hiện `requested_slots`.

Chuẩn hóa giá trị (v2.2):
- Áp dụng lowercasing nhất quán, chuẩn thời gian/ngày, bỏ khoảng trắng thừa, ánh xạ đồng nghĩa; slot đóng so khớp theo giá trị canonical.

### 7) Gợi ý thực hành
- Dùng cùng pipeline tiền xử lý/chuẩn hóa như dữ liệu v2.2 khi huấn luyện và đánh giá.
- Với slot có thể có nhiều giá trị, cần dự đoán đầy đủ và đúng thứ tự không quan trọng, nhưng tập giá trị phải trùng sau chuẩn hóa.
- Ưu tiên tận dụng `state_update` (nếu có) để huấn luyện mô hình cập nhật gia tăng, giúp ổn định và hiệu quả hơn.

---

Tóm lại: Bản gốc (2.0) giàu dữ liệu nhưng nhiễu nhãn; 2.1 sửa lỗi một phần; 2.2 tái-chuẩn hóa sâu, phẳng hóa `domain-slot`, chuẩn hóa giá trị và nhãn, giúp đánh giá DST trở nên đáng tin cậy hơn (JGA, Slot/Requested F1). Thư mục hiện tại đã theo bố cục train/dev/test của MultiWOZ, kèm công cụ chuyển đổi/chuẩn hóa để làm việc thuận tiện với định dạng v2.2.

