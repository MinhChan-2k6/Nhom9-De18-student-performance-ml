"""Cấu hình dùng chung cho giao diện Streamlit."""

from __future__ import annotations

import os

from dotenv import load_dotenv


# Đọc cấu hình từ file .env khi chạy ở máy local.
load_dotenv()

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"

# Khi chạy Docker, giá trị này có thể được đặt thành http://api:8000.
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    DEFAULT_API_BASE_URL,
).strip().rstrip("/")

if not API_BASE_URL:
    API_BASE_URL = DEFAULT_API_BASE_URL
