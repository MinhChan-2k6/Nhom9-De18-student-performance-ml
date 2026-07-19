from fastapi import APIRouter

from api.database import check_database_connection
from api.schemas import HealthResponse


router = APIRouter(
    prefix="/health",
    tags=["System"],
)


@router.get(
    "",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """
    Kiểm tra trạng thái hoạt động của API và kết nối SQL Server.
    """

    database_connected = check_database_connection()

    return HealthResponse(
        status="ok",
        api="online",
        database=(
            "connected"
            if database_connected
            else "disconnected"
        ),
    )