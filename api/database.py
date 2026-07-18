from __future__ import annotations

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker


# Đọc các biến môi trường từ file .env
load_dotenv()


# ============================================================
# DATABASE CONFIGURATION
# Không hard-code tên SQL Server của máy cá nhân trong source code
# ============================================================

DB_SERVER = os.getenv(
    "DB_SERVER",
    "localhost",
)

DB_NAME = os.getenv(
    "DB_NAME",
    "StudentPerformanceDB",
)

DB_DRIVER = os.getenv(
    "DB_DRIVER",
    "ODBC Driver 17 for SQL Server",
)

DB_TRUSTED_CONNECTION = os.getenv(
    "DB_TRUSTED_CONNECTION",
    "yes",
)


# ============================================================
# SQL SERVER CONNECTION STRING
# ============================================================

connection_string = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    f"Trusted_Connection={DB_TRUSTED_CONNECTION};"
    "TrustServerCertificate=yes;"
)


# Tạo URL kết nối cho SQLAlchemy + pyodbc
database_url = URL.create(
    "mssql+pyodbc",
    query={
        "odbc_connect": connection_string,
    },
)


# ============================================================
# SQLALCHEMY ENGINE
# ============================================================

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    fast_executemany=True,
)


# ============================================================
# DATABASE SESSION
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency cung cấp database session cho FastAPI.

    Session sẽ tự động được đóng sau khi request hoàn thành.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


def check_database_connection() -> bool:
    """
    Kiểm tra kết nối đến SQL Server.

    Trả về:
        True  - Kết nối thành công
        False - Không thể kết nối
    """

    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        return True

    except Exception:
        return False