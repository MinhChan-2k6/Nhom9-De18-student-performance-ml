from __future__ import annotations

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker


load_dotenv()


DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "StudentPerformanceDB")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_USERNAME = os.getenv("DB_USER") or os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD") or os.getenv("DB_PASS")
DB_AUTH_MODE = os.getenv("DB_AUTH_MODE", "").strip().lower()
DB_ENCRYPT = os.getenv("DB_ENCRYPT", "yes")
DB_TRUST_SERVER_CERTIFICATE = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "yes")
DB_CONNECTION_TIMEOUT = os.getenv("DB_CONNECTION_TIMEOUT", "5")


WINDOWS_AUTH_MODES = {"windows", "trusted", "integrated", "windows_auth"}
SQL_AUTH_MODES = {"sql", "sqlserver", "sql_auth", "password"}


def resolve_auth_mode() -> str:

    if DB_AUTH_MODE:
        normalized_mode = DB_AUTH_MODE
    elif DB_USERNAME or DB_PASSWORD:
        normalized_mode = "sql"
    else:
        normalized_mode = "windows"

    if normalized_mode in WINDOWS_AUTH_MODES:
        return "windows"

    if normalized_mode in SQL_AUTH_MODES:
        return "sql"

    valid_modes = ", ".join(sorted(WINDOWS_AUTH_MODES | SQL_AUTH_MODES))
    raise ValueError(
        f"Không xác định DB_AUTH_MODE={DB_AUTH_MODE!r}. Các giá trị hợp lệ: {valid_modes}."
    )


def build_connection_string() -> str:

    parts = [
        f"DRIVER={{{DB_DRIVER}}}",
        f"SERVER={DB_SERVER}",
        f"DATABASE={DB_NAME}",
        f"Encrypt={DB_ENCRYPT}",
        f"TrustServerCertificate={DB_TRUST_SERVER_CERTIFICATE}",
        f"Connection Timeout={DB_CONNECTION_TIMEOUT}",
    ]

    auth_mode = resolve_auth_mode()
    if auth_mode == "windows":
        parts.append("Trusted_Connection=yes")
    else:
        if not DB_USERNAME or not DB_PASSWORD:
            raise ValueError(
                "SQL authentication requires DB_USER/DB_USERNAME and DB_PASSWORD."
            )
        parts.extend(
            [
                f"UID={DB_USERNAME}",
                f"PWD={DB_PASSWORD}",
            ]
        )

    return ";".join(parts) + ";"


connection_string = build_connection_string()

database_url = URL.create(
    "mssql+pyodbc",
    query={"odbc_connect": connection_string},
)

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    fast_executemany=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> bool:

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
