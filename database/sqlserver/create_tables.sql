/* =========================================================
   STUDENT PERFORMANCE PREDICTION
   DATABASE INITIALIZATION - SQL SERVER

   Bài toán: Hồi quy dự đoán điểm G3
   Không phân loại Đạt / Trượt
   ========================================================= */


/* =========================================================
   1. CREATE DATABASE
   ========================================================= */

IF DB_ID(N'StudentPerformanceDB') IS NULL
BEGIN
    CREATE DATABASE StudentPerformanceDB;
END;
GO


USE StudentPerformanceDB;
GO


/* =========================================================
   2. TABLE: PredictionHistory
   Lưu lịch sử dự đoán điểm G3
   ========================================================= */

IF OBJECT_ID(
    N'dbo.PredictionHistory',
    N'U'
) IS NULL
BEGIN
    CREATE TABLE dbo.PredictionHistory
    (
        prediction_id INT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_PredictionHistory
            PRIMARY KEY,

        student_code NVARCHAR(50) NULL,

        full_name NVARCHAR(100) NULL,

        subject NVARCHAR(50) NULL,

        prediction_mode NVARCHAR(50) NOT NULL,

        scenario NVARCHAR(50) NULL,

        predicted_g3 DECIMAL(6,2) NOT NULL,

        model_name NVARCHAR(100) NOT NULL,

        model_version NVARCHAR(30) NULL,

        input_json NVARCHAR(MAX) NULL,

        created_at DATETIME2 NOT NULL
            CONSTRAINT DF_PredictionHistory_CreatedAt
            DEFAULT SYSDATETIME()
    );
END;
GO


/* =========================================================
   3. REMOVE OLD CLASSIFICATION COLUMN
   Hỗ trợ database cũ đã từng có Đạt / Trượt
   ========================================================= */

IF EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_PredictionHistory_Result'
)
BEGIN
    ALTER TABLE dbo.PredictionHistory
    DROP CONSTRAINT CK_PredictionHistory_Result;
END;
GO


IF COL_LENGTH(
    'dbo.PredictionHistory',
    'predicted_result'
) IS NOT NULL
BEGIN
    ALTER TABLE dbo.PredictionHistory
    DROP COLUMN predicted_result;
END;
GO


/* =========================================================
   4. TABLE: ModelVersions
   Lưu thông tin các phiên bản model hồi quy
   ========================================================= */

IF OBJECT_ID(
    N'dbo.ModelVersions',
    N'U'
) IS NULL
BEGIN
    CREATE TABLE dbo.ModelVersions
    (
        model_version_id INT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ModelVersions
            PRIMARY KEY,

        model_name NVARCHAR(100) NOT NULL,

        version NVARCHAR(30) NULL,

        algorithm NVARCHAR(100) NULL,

        scenario NVARCHAR(50) NULL,

        mae FLOAT NULL,

        rmse FLOAT NULL,

        r2_score FLOAT NULL,

        artifact_path NVARCHAR(500) NULL,

        is_active BIT NOT NULL
            CONSTRAINT DF_ModelVersions_IsActive
            DEFAULT 1,

        created_at DATETIME2 NOT NULL
            CONSTRAINT DF_ModelVersions_CreatedAt
            DEFAULT SYSDATETIME(),

        model_version NVARCHAR(30) NULL,

        metrics_json NVARCHAR(MAX) NULL
    );
END;
GO


/* =========================================================
   5. ADD MISSING COLUMNS FOR OLD DATABASES
   ========================================================= */

IF COL_LENGTH(
    'dbo.ModelVersions',
    'version'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD version NVARCHAR(30) NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'algorithm'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD algorithm NVARCHAR(100) NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'scenario'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD scenario NVARCHAR(50) NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'mae'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD mae FLOAT NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'rmse'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD rmse FLOAT NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'r2_score'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD r2_score FLOAT NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'artifact_path'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD artifact_path NVARCHAR(500) NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'is_active'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD is_active BIT NOT NULL
        CONSTRAINT DF_ModelVersions_IsActive_Migration
        DEFAULT 1;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'created_at'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD created_at DATETIME2 NOT NULL
        CONSTRAINT DF_ModelVersions_CreatedAt_Migration
        DEFAULT SYSDATETIME();
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'model_version'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD model_version NVARCHAR(30) NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'metrics_json'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD metrics_json NVARCHAR(MAX) NULL;
END;
GO


/* =========================================================
   6. INDEXES
   ========================================================= */

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_PredictionHistory_CreatedAt'
      AND object_id = OBJECT_ID(
          N'dbo.PredictionHistory'
      )
)
BEGIN
    CREATE INDEX IX_PredictionHistory_CreatedAt
    ON dbo.PredictionHistory(created_at DESC);
END;
GO


IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_PredictionHistory_ModeSubject'
      AND object_id = OBJECT_ID(
          N'dbo.PredictionHistory'
      )
)
BEGIN
    CREATE INDEX IX_PredictionHistory_ModeSubject
    ON dbo.PredictionHistory(
        prediction_mode,
        subject
    );
END;
GO


IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_ModelVersions_Scenario'
      AND object_id = OBJECT_ID(
          N'dbo.ModelVersions'
      )
)
BEGIN
    CREATE INDEX IX_ModelVersions_Scenario
    ON dbo.ModelVersions(scenario);
END;
GO


PRINT N'Khởi tạo database cho bài toán hồi quy thành công.';
GO