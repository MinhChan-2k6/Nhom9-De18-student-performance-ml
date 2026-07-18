USE StudentPerformanceDB;
GO


/* =========================================================
   VIEW 1: Prediction History Detail
   Chi tiết lịch sử dự đoán hồi quy
   ========================================================= */

CREATE OR ALTER VIEW dbo.vw_PredictionHistoryDetail
AS
SELECT
    prediction_id,
    student_code,
    full_name,
    subject,
    prediction_mode,
    scenario,
    predicted_g3,
    model_name,
    model_version,
    input_json,
    created_at
FROM dbo.PredictionHistory;
GO


/* =========================================================
   VIEW 2: Prediction Summary
   Thống kê tổng quan kết quả hồi quy
   ========================================================= */

CREATE OR ALTER VIEW dbo.vw_PredictionSummary
AS
SELECT
    COUNT(*) AS total_predictions,

    CAST(
        AVG(
            CAST(predicted_g3 AS FLOAT)
        )
        AS DECIMAL(6,2)
    ) AS average_g3,

    MIN(predicted_g3) AS minimum_g3,

    MAX(predicted_g3) AS maximum_g3

FROM dbo.PredictionHistory;
GO


/* =========================================================
   VIEW 3: Active Model Versions
   Danh sách model hồi quy đang hoạt động
   ========================================================= */

CREATE OR ALTER VIEW dbo.vw_ActiveModelVersions
AS
SELECT
    model_version_id,
    model_name,
    version,
    algorithm,
    scenario,
    mae,
    rmse,
    r2_score,
    artifact_path,
    is_active,
    model_version,
    metrics_json,
    created_at
FROM dbo.ModelVersions
WHERE is_active = 1;
GO


PRINT N'Đã tạo/cập nhật các VIEW cho bài toán hồi quy.';
GO