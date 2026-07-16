# Model Card — Student Performance Regression

## 1. Bài toán

- Loại bài toán: Hồi quy
- Biến mục tiêu: `G3`
- Thang điểm mục tiêu: 0–20
- Model được chọn: `random_forest_grid_search`
- Scenario được chọn: `A_with_G1_G2`

## 2. Scenario

- `A_with_G1_G2`: sử dụng G1 và G2.
- `B_without_G1_G2`: không sử dụng G1, G2; phục vụ dự báo sớm.

## 3. Chỉ số trên tập test

- MAE: `0.925801`
- RMSE: `1.580415`
- R²: `0.831716`

## 4. Số lượng đặc trưng đầu vào

- Số cột đầu vào trước tiền xử lý: `33`

## 5. Lưu ý

- MAE và RMSE càng thấp càng tốt.
- R² càng gần 1 càng tốt.
- Kết quả Đạt/Trượt chỉ là quy đổi từ điểm G3 dự đoán.
- Model chính vẫn là mô hình hồi quy.
- Scenario có G1, G2 thường chính xác hơn nhưng chỉ dùng được khi đã có điểm trước đó.
- Scenario không có G1, G2 phù hợp cho cảnh báo sớm nhưng có thể kém chính xác hơn.

## 6. Giới hạn

- Dữ liệu gốc sử dụng thang điểm 0–20.
- Model có thể không khái quát tốt cho mọi trường học hoặc hệ thống giáo dục.
- Không nên sử dụng dự đoán như quyết định duy nhất về kết quả học tập.
