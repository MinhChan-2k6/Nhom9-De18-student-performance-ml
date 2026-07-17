# Model Card — Student Performance Regression

## 1. Bài toán

- Loại bài toán: Hồi quy
- Biến mục tiêu: `G3`
- Thang điểm mục tiêu: 0–20
- Model được chọn: `random_forest`
- Scenario: `A_with_G1_G2`
- Cách dùng đặc trưng: Có sử dụng `G1`, `G2`.

## 2. Chỉ số trên tập test

- MAE: `0.976983`
- RMSE: `1.450978`
- R²: `0.854988`
- Accuracy Đạt/Trượt: `0.810127`
- F1 Đạt/Trượt: `0.867257`
- Ngưỡng Đạt: `G3 >= 10.0`

## 3. Đặc trưng và output

- Số cột đầu vào trước tiền xử lý: `33`
- Danh sách đầu vào: `subject, school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2`
- Số biểu đồ đánh giá: `10`
- Model: `artifacts/Model_Have_A_B/best_regression_model.pkl`
- Metrics: `artifacts/Model_Have_A_B/metrics.json`
- Biểu đồ: `artifacts/Model_Have_A_B/plots`

## 4. Lưu ý

- MAE và RMSE càng thấp càng tốt; R² càng gần 1 càng tốt.
- Đạt/Trượt chỉ là kết quả quy đổi từ điểm G3 dự đoán.
- Không nên dùng dự đoán như quyết định duy nhất về kết quả học tập.
