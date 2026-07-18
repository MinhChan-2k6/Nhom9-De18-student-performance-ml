/* =========================================================
   DATABASE: StudentPerformanceDB
   Mục đích:
   - Tạo database nếu chưa tồn tại
   - Tạo bảng PredictionHistory
   - Tạo bảng ModelVersions
   - Bổ sung cột còn thiếu nếu database đã tồn tại
   - Tạo index phục vụ truy vấn
   ========================================================= */

USE master;
GO

IF DB_ID(N'StudentPerformanceDB') IS NULL
BEGIN
    CREATE DATABASE StudentPerformanceDB;
END;
GO


USE StudentPerformanceDB;
GO


/* =========================================================
   1. BẢNG LỊCH SỬ DỰ ĐOÁN
   ========================================================= */

IF OBJECT_ID(N'dbo.PredictionHistory', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.PredictionHistory (
        prediction_id INT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_PredictionHistory PRIMARY KEY,

        student_code NVARCHAR(50) NULL,

        full_name NVARCHAR(100) NULL,

        subject NVARCHAR(50) NULL,

        prediction_mode NVARCHAR(50) NOT NULL,

        scenario NVARCHAR(50) NULL,

        predicted_g3 DECIMAL(5,2) NOT NULL,

        predicted_result NVARCHAR(20) NOT NULL,

        model_name NVARCHAR(100) NOT NULL,

        model_version NVARCHAR(30) NULL,

        input_json NVARCHAR(MAX) NULL,

        created_at DATETIME2(0) NOT NULL
            CONSTRAINT DF_PredictionHistory_CreatedAt
            DEFAULT SYSDATETIME(),

        CONSTRAINT CK_PredictionHistory_G3
            CHECK (
                predicted_g3 >= 0
                AND predicted_g3 <= 20
            ),

        CONSTRAINT CK_PredictionHistory_Mode
            CHECK (
                prediction_mode IN (
                    N'with_g1_g2',
                    N'early_prediction'
                )
            ),

        CONSTRAINT CK_PredictionHistory_Result
            CHECK (
                predicted_result IN (
                    N'Đạt',
                    N'Trượt'
                )
            ),

        CONSTRAINT CK_PredictionHistory_Subject
            CHECK (
                subject IS NULL
                OR subject IN (
                    N'math',
                    N'portuguese'
                )
            ),

        CONSTRAINT CK_PredictionHistory_InputJson
            CHECK (
                input_json IS NULL
                OR ISJSON(input_json) = 1
            )
    );
END;
GO


/* =========================================================
   2. BẢNG QUẢN LÝ PHIÊN BẢN MODEL

   Cấu trúc này khớp với database hiện tại của nhóm:
   - model_version_id
   - model_name
   - version
   - algorithm
   - scenario
   - mae
   - rmse
   - r2_score
   - artifact_path
   - is_active
   - created_at
   - model_version
   - metrics_json
   ========================================================= */

IF OBJECT_ID(N'dbo.ModelVersions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ModelVersions (
        model_version_id INT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ModelVersions PRIMARY KEY,

        model_name NVARCHAR(100) NOT NULL,

        version NVARCHAR(30) NULL,

        algorithm NVARCHAR(100) NULL,

        scenario NVARCHAR(50) NOT NULL,

        mae DECIMAL(10,6) NULL,

        rmse DECIMAL(10,6) NULL,

        r2_score DECIMAL(10,6) NULL,

        artifact_path NVARCHAR(260) NULL,

        is_active BIT NOT NULL
            CONSTRAINT DF_ModelVersions_IsActive
            DEFAULT 1,

        created_at DATETIME2(0) NOT NULL
            CONSTRAINT DF_ModelVersions_CreatedAt
            DEFAULT SYSDATETIME(),

        model_version NVARCHAR(30) NULL,

        metrics_json NVARCHAR(MAX) NULL
    );
END;
GO


/* =========================================================
   3. BỔ SUNG CỘT NẾU BẢNG MODELVERSIONS ĐÃ TỒN TẠI
   Giúp script chạy được với database cũ của nhóm
   ========================================================= */

IF COL_LENGTH(
    'dbo.ModelVersions',
    'model_name'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD model_name NVARCHAR(100) NULL;
END;
GO


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
    ADD mae DECIMAL(10,6) NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'rmse'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD rmse DECIMAL(10,6) NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'r2_score'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD r2_score DECIMAL(10,6) NULL;
END;
GO


IF COL_LENGTH(
    'dbo.ModelVersions',
    'artifact_path'
) IS NULL
BEGIN
    ALTER TABLE dbo.ModelVersions
    ADD artifact_path NVARCHAR(260) NULL;
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
    ADD created_at DATETIME2(0) NOT NULL
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
   4. INDEX CHO PREDICTION HISTORY
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
        ON dbo.PredictionHistory(
            created_at DESC
        );
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
        )
        INCLUDE (
            predicted_g3,
            predicted_result,
            created_at
        );
END;
GO


/* =========================================================
   5. INDEX CHO MODEL VERSIONS
   ========================================================= */

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
        ON dbo.ModelVersions(
            scenario,
            is_active
        );
END;
GO


/* =========================================================
   HOÀN TẤT
   ========================================================= */

PRINT N'Khởi tạo cấu trúc StudentPerformanceDB thành công.';
GO