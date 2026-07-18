# Model Card — Student Performance Regression

## 1. Bài toán

- Loại bài toán: Hồi quy
- Biến mục tiêu: `G3`
- Thang điểm mục tiêu: 0–20
- Model được chọn: `random_forest_grid_search`
- Scenario: `B_without_G1_G2`
- Feature set được chọn: `raw`
- Cách dùng đặc trưng: Không sử dụng `G1`, `G2`; phù hợp cho dự báo sớm.

## 2. Chỉ số trên tập test

- MAE: `2.12298`
- RMSE: `2.775914`
- R²: `0.469244`

## 3. So sánh feature engineering trên validation

- `raw`: model tốt nhất `random_forest_grid_search`, RMSE=`3.892388`, MAE=`2.866632`, R²=`0.246147`
- `engineered`: model tốt nhất `random_forest_grid_search`, RMSE=`3.895362`, MAE=`2.870218`, R²=`0.244994`

- Đặc trưng dẫn xuất được model cuối sử dụng: `Không có`


## 4. Tinh chỉnh siêu tham số của model được chọn

- Phương pháp: `GridSearchCV`
- Chiến lược CV: `GroupKFold`
- Cách nhóm: `student_identity`
- Số nhóm học sinh trong train: `463`
- Số fold: `3`
- Số cấu hình: `16`
- Tổng lượt fit: `48`
- CV RMSE tốt nhất: `3.210404`
- Tham số tốt nhất: `model__max_depth`=10, `model__max_features`=1.0, `model__min_samples_leaf`=1, `model__min_samples_split`=5, `model__n_estimators`=300


## 5. Đặc trưng và output

- Số cột đầu vào thô: `31`
- Số cột trước tiền xử lý sau feature engineering: `31`
- Danh sách đầu vào thô: `subject, school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences`
- Đầu ra của model: Điểm `G3` dạng số thực
- Số biểu đồ đánh giá: `9`
- Model: `artifacts/Model_Dont_Have_A_B/best_regression_model.pkl`
- Metrics: `artifacts/Model_Dont_Have_A_B/metrics.json`
- Biểu đồ: `artifacts/Model_Dont_Have_A_B/plots`

## 6. Lưu ý

- GroupKFold giữ các bản ghi của cùng học sinh trong cùng một fold khi tuning.
- MAE và RMSE càng thấp càng tốt; R² càng gần 1 càng tốt.
- Đầu ra duy nhất của hệ thống là điểm G3 dự đoán trên thang 0–20.
- Không nên dùng dự đoán như quyết định duy nhất về kết quả học tập.
