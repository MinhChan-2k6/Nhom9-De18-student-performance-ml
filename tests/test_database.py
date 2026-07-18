from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATABASE_ENV_VARS = [
    "DB_SERVER",
    "DB_NAME",
    "DB_DRIVER",
    "DB_USER",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_PASS",
    "DB_AUTH_MODE",
    "DB_ENCRYPT",
    "DB_TRUST_SERVER_CERTIFICATE",
    "DB_CONNECTION_TIMEOUT",
]


def reload_database_module(monkeypatch: pytest.MonkeyPatch, **env_values):
    """Reload api.database after applying an isolated environment."""

    for name in DATABASE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    for name, value in env_values.items():
        monkeypatch.setenv(name, value)

    sys.modules.pop("api.database", None)
    return importlib.import_module("api.database")


def test_default_configuration_uses_windows_authentication(monkeypatch):
    database = reload_database_module(monkeypatch)

    assert database.resolve_auth_mode() == "windows"
    assert "DRIVER={ODBC Driver 17 for SQL Server}" in database.connection_string
    assert "SERVER=localhost" in database.connection_string
    assert "DATABASE=StudentPerformanceDB" in database.connection_string
    assert "Trusted_Connection=yes" in database.connection_string
    assert "UID=" not in database.connection_string
    assert "PWD=" not in database.connection_string


def test_explicit_windows_authentication_mode(monkeypatch):
    database = reload_database_module(
        monkeypatch,
        DB_AUTH_MODE="trusted",
        DB_SERVER="localhost\\SQLEXPRESS",
    )

    assert database.resolve_auth_mode() == "windows"
    assert "SERVER=localhost\\SQLEXPRESS" in database.connection_string
    assert "Trusted_Connection=yes" in database.connection_string


def test_sql_authentication_for_docker_configuration(monkeypatch):
    database = reload_database_module(
        monkeypatch,
        DB_AUTH_MODE="sql",
        DB_SERVER="db,1433",
        DB_NAME="StudentPerformanceDB",
        DB_DRIVER="ODBC Driver 18 for SQL Server",
        DB_USER="sa",
        DB_PASSWORD="StrongPassword123!",
        DB_ENCRYPT="yes",
        DB_TRUST_SERVER_CERTIFICATE="yes",
        DB_CONNECTION_TIMEOUT="10",
    )

    assert database.resolve_auth_mode() == "sql"
    assert "DRIVER={ODBC Driver 18 for SQL Server}" in database.connection_string
    assert "SERVER=db,1433" in database.connection_string
    assert "DATABASE=StudentPerformanceDB" in database.connection_string
    assert "UID=sa" in database.connection_string
    assert "PWD=StrongPassword123!" in database.connection_string
    assert "Encrypt=yes" in database.connection_string
    assert "TrustServerCertificate=yes" in database.connection_string
    assert "Connection Timeout=10" in database.connection_string
    assert "Trusted_Connection=yes" not in database.connection_string


def test_sql_authentication_is_inferred_when_credentials_exist(monkeypatch):
    database = reload_database_module(
        monkeypatch,
        DB_USERNAME="student_user",
        DB_PASS="student_password",
    )

    assert database.resolve_auth_mode() == "sql"
    assert "UID=student_user" in database.connection_string
    assert "PWD=student_password" in database.connection_string


def test_sql_authentication_requires_username_and_password(monkeypatch):
    with pytest.raises(ValueError, match="SQL authentication requires"):
        reload_database_module(
            monkeypatch,
            DB_AUTH_MODE="sql",
            DB_USER="sa",
        )


@pytest.mark.parametrize(
    "invalid_mode",
    ["kerberos", "oauth", "none"],
)
def test_invalid_authentication_mode_is_rejected(monkeypatch, invalid_mode):
    with pytest.raises(ValueError, match="DB_AUTH_MODE"):
        reload_database_module(
            monkeypatch,
            DB_AUTH_MODE=invalid_mode,
        )


def test_check_database_connection_returns_true_when_query_succeeds(monkeypatch):
    database = reload_database_module(monkeypatch)

    class WorkingConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, query):
            assert str(query) == "SELECT 1"

    class WorkingEngine:
        def connect(self):
            return WorkingConnection()

    monkeypatch.setattr(database, "engine", WorkingEngine())

    assert database.check_database_connection() is True


def test_check_database_connection_returns_false_when_query_fails(monkeypatch):
    database = reload_database_module(monkeypatch)

    class FailingEngine:
        def connect(self):
            raise RuntimeError("Không kết nối được database")

    monkeypatch.setattr(database, "engine", FailingEngine())

    assert database.check_database_connection() is False
