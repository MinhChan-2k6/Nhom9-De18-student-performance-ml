USE StudentPerformanceDB;
GO

/* ============================================
   MODEL 1 - Có G1 và G2
   ============================================ */
IF NOT EXISTS (
    SELECT 1
    FROM dbo.ModelVersions
    WHERE scenario = N'A_with_G1_G2'
      AND model_version = N'1.0.0'
)
BEGIN
    INSERT INTO dbo.ModelVersions (
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
        metrics_json
    )
    VALUES (
        N'random_forest_grid_search',
        N'1.0.0',
        N'Random Forest + GridSearchCV',
        N'A_with_G1_G2',
        0.956767,
        1.389162,
        0.867080,
        N'artifacts\Model_Have_A_B\best_regression_model.pkl',
        1,
        N'1.0.0',
        N'{"scenario":"A_with_G1_G2","source":"metrics.json"}'
    );
END;
GO


/* ============================================
   MODEL 2 - Không có G1 và G2
   ============================================ */
IF NOT EXISTS (
    SELECT 1
    FROM dbo.ModelVersions
    WHERE scenario = N'B_without_G1_G2'
      AND model_version = N'1.0.0'
)
BEGIN
    INSERT INTO dbo.ModelVersions (
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
        metrics_json
    )
    VALUES (
        N'random_forest',
        N'1.0.0',
        N'Random Forest',
        N'B_without_G1_G2',
        2.122980,
        2.775914,
        0.469244,
        N'artifacts\Model_Dont_Have_A_B\best_regression_model.pkl',
        1,
        N'1.0.0',
        N'{"scenario":"B_without_G1_G2","source":"metrics.json"}'
    );
END;
GO