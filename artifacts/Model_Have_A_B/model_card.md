# Model Card — Student Performance Regression

## 1. Bài toán

- Loại bài toán: Hồi quy
- Biến mục tiêu: `G3`
- Thang điểm mục tiêu: 0–20
- Model được chọn: `random_forest_grid_search`
- Scenario: `A_with_G1_G2`
- Feature set được chọn: `engineered`
- Cách dùng đặc trưng: Có sử dụng `G1`, `G2`.

## 2. Chỉ số trên tập test

- MAE: `0.956767`
- RMSE: `1.389162`
- R²: `0.86708`

## 3. So sánh feature engineering trên validation

- `raw`: model tốt nhất `random_forest_grid_search`, RMSE=`1.893636`, MAE=`1.140948`, R²=`0.821578`
- `engineered`: model tốt nhất `random_forest_grid_search`, RMSE=`1.866449`, MAE=`1.123748`, R²=`0.826665`

- Đặc trưng dẫn xuất được model cuối sử dụng: `parent_education_mean, alcohol_consumption_mean, social_activity_score, support_count, absence_log1p, previous_grade_mean, grade_trend`


## 4. Tinh chỉnh siêu tham số của model được chọn

- Phương pháp: `GridSearchCV`
- Chiến lược CV: `GroupKFold`
- Cách nhóm: `student_identity`
- Số nhóm học sinh trong train: `463`
- Số fold: `3`
- Số cấu hình: `16`
- Tổng lượt fit: `48`
- CV RMSE tốt nhất: `1.368974`
- Tham số tốt nhất: `model__max_depth`=10, `model__max_features`=1.0, `model__min_samples_leaf`=1, `model__min_samples_split`=5, `model__n_estimators`=300


## 5. Đặc trưng và output

- Số cột đầu vào thô: `33`
- Số cột trước tiền xử lý sau feature engineering: `40`
- Danh sách đầu vào thô: `subject, school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2`
- Đầu ra của model: Điểm `G3` dạng số thực
- Số biểu đồ đánh giá: `9`
- Model: `artifacts/Model_Have_A_B/best_regression_model.pkl`
- Metrics: `artifacts/Model_Have_A_B/metrics.json`
- Biểu đồ: `artifacts/Model_Have_A_B/plots`

## 6. Lưu ý

- GroupKFold giữ các bản ghi của cùng học sinh trong cùng một fold khi tuning.
- MAE và RMSE càng thấp càng tốt; R² càng gần 1 càng tốt.
- Đầu ra duy nhất của hệ thống là điểm G3 dự đoán trên thang 0–20.
- Không nên dùng dự đoán như quyết định duy nhất về kết quả học tập.
