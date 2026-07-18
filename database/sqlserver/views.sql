USE StudentPerformanceDB;
GO

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
    predicted_result,
    model_name,
    model_version,
    input_json,
    created_at
FROM dbo.PredictionHistory;
GO


CREATE OR ALTER VIEW dbo.vw_PredictionSummary
AS
SELECT
    COUNT(*) AS total_predictions,
    AVG(CAST(predicted_g3 AS DECIMAL(10,2))) AS average_g3,
    MIN(predicted_g3) AS minimum_g3,
    MAX(predicted_g3) AS maximum_g3,
    SUM(
        CASE
            WHEN predicted_result = N'Đạt' THEN 1
            ELSE 0
        END
    ) AS total_passed,
    SUM(
        CASE
            WHEN predicted_result = N'Trượt' THEN 1
            ELSE 0
        END
    ) AS total_failed
FROM dbo.PredictionHistory;
GO


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
    created_at,
    model_version,
    metrics_json
FROM dbo.ModelVersions
WHERE is_active = 1;
GO