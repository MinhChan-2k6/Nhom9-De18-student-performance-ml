from fastapi import APIRouter

from api.database import check_database_connection


router = APIRouter(
    prefix="/health",
    tags=["System"],
)


@router.get("")
def health_check() -> dict:
    """
    Kiểm tra trạng thái hoạt động của API và kết nối SQL Server.
    """

    database_connected = check_database_connection()

    return {
        "status": "ok",
        "api": "online",
        "database": (
            "connected"
            if database_connected
            else "disconnected"
        ),
    }