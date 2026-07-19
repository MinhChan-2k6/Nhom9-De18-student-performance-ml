USE StudentPerformanceDB;
GO

IF EXISTS (
    SELECT 1
    FROM dbo.ModelVersions
    WHERE scenario = N'A_with_G1_G2'
      AND (model_version = N'1.0.0' OR version = N'1.0.0')
)
BEGIN
    UPDATE dbo.ModelVersions
    SET
        model_name = N'random_forest_grid_search',
        version = N'1.0.0',
        model_version = N'1.0.0',
        algorithm = N'Random Forest + GridSearchCV',
        mae = 0.956767,
        rmse = 1.389162,
        r2_score = 0.867080,
        artifact_path = N'artifacts\Model_Have_A_B\best_regression_model.pkl',
        is_active = 1,
        metrics_json = N'{"scenario":"A_with_G1_G2","best_model":"random_forest_grid_search","source":"metrics.json"}'
    WHERE scenario = N'A_with_G1_G2'
      AND (model_version = N'1.0.0' OR version = N'1.0.0');
END
ELSE
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
        N'{"scenario":"A_with_G1_G2","best_model":"random_forest_grid_search","source":"metrics.json"}'
    );
END;
GO


IF EXISTS (
    SELECT 1
    FROM dbo.ModelVersions
    WHERE scenario = N'B_without_G1_G2'
      AND (model_version = N'1.0.0' OR version = N'1.0.0')
)
BEGIN
    UPDATE dbo.ModelVersions
    SET
        model_name = N'random_forest_grid_search',
        version = N'1.0.0',
        model_version = N'1.0.0',
        algorithm = N'Random Forest + GridSearchCV',
        mae = 2.122980,
        rmse = 2.775914,
        r2_score = 0.469244,
        artifact_path = N'artifacts\Model_Dont_Have_A_B\best_regression_model.pkl',
        is_active = 1,
        metrics_json = N'{"scenario":"B_without_G1_G2","best_model":"random_forest_grid_search","source":"metrics.json"}'
    WHERE scenario = N'B_without_G1_G2'
      AND (model_version = N'1.0.0' OR version = N'1.0.0');
END
ELSE
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
        N'B_without_G1_G2',
        2.122980,
        2.775914,
        0.469244,
        N'artifacts\Model_Dont_Have_A_B\best_regression_model.pkl',
        1,
        N'1.0.0',
        N'{"scenario":"B_without_G1_G2","best_model":"random_forest_grid_search","source":"metrics.json"}'
    );
END;
GO
