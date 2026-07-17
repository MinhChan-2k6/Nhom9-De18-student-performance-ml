# Model Card — Student Performance Regression

## 1. Bài toán

- Loại bài toán: Hồi quy
- Biến mục tiêu: `G3`
- Thang điểm mục tiêu: 0–20
- Model được chọn: `random_forest_grid_search`
- Scenario: `A_with_G1_G2`
- Cách dùng đặc trưng: Có sử dụng `G1`, `G2`.

## 2. Chỉ số trên tập test

- MAE: `0.966044`
- RMSE: `1.425448`
- R²: `0.860046`


## 3. Tinh chỉnh siêu tham số

- Phương pháp: `GridSearchCV`
- Số fold: `3`
- Số cấu hình: `16`
- Tổng lượt fit: `48`
- CV RMSE tốt nhất: `1.594842`
- Tham số tốt nhất: `model__max_depth`=None, `model__max_features`=1.0, `model__min_samples_leaf`=1, `model__min_samples_split`=2, `model__n_estimators`=200


## 4. Đặc trưng và output

- Số cột đầu vào trước tiền xử lý: `33`
- Danh sách đầu vào: `subject, school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2`
- Đầu ra của model: Điểm `G3` dạng số thực
- Số biểu đồ đánh giá: `9`
- Model: `artifacts/Model_Have_A_B/best_regression_model.pkl`
- Metrics: `artifacts/Model_Have_A_B/metrics.json`
- Biểu đồ: `artifacts/Model_Have_A_B/plots`

## 5. Lưu ý

- MAE và RMSE càng thấp càng tốt; R² càng gần 1 càng tốt.
- Đầu ra duy nhất của hệ thống là điểm G3 dự đoán trên thang 0–20.
- Không nên dùng dự đoán như quyết định duy nhất về kết quả học tập.
