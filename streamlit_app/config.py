"""Cấu hình dùng chung cho ứng dụng Streamlit."""

from __future__ import annotations

import os

from dotenv import load_dotenv


# ==================================================
# ĐỌC BIẾN MÔI TRƯỜNG
# ==================================================
# Đọc cấu hình từ file .env khi chạy trên máy local.
load_dotenv()


# ==================================================
# FASTAPI
# ==================================================
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"

# Khi chạy Docker có thể cấu hình:
# API_BASE_URL=http://api:8000
API_BASE_URL = (
    os.getenv(
        "API_BASE_URL",
        DEFAULT_API_BASE_URL,
    )
    .strip()
    .rstrip("/")
)

# Nếu biến môi trường rỗng thì dùng địa chỉ mặc định.
if not API_BASE_URL:
    API_BASE_URL = DEFAULT_API_BASE_URL