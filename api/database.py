from __future__ import annotations

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DB_SERVER = os.getenv("DB_SERVER", "LTHANH")
DB_NAME = os.getenv("DB_NAME", "StudentPerformanceDB")
DB_DRIVER = os.getenv(
    "DB_DRIVER",
    "ODBC Driver 17 for SQL Server",
)

connection_string = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

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
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return True