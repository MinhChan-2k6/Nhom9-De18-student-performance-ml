from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from streamlit_app.components.styles import load_css


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Lịch sử dự đoán",
    page_icon="📜",
    layout="wide",
)

load_css()

st.title("📜 Lịch sử dự đoán")
st.caption("Tra cứu các lần dự đoán đã được lưu trong SQL Server.")


def load_prediction_history(limit: int = 500) -> list[dict]:
    """Lấy lịch sử dự đoán từ FastAPI."""

    response = requests.get(
        f"{API_BASE_URL}/history",
        params={"limit": limit},
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError("Dữ liệu lịch sử từ API không đúng định dạng.")

    return data


def subject_to_vietnamese(subject: str | None) -> str:
    mapping = {
        "math": "Toán",
        "portuguese": "Tiếng Bồ Đào Nha",
    }

    return mapping.get(subject, subject or "Không xác định")


def mode_to_vietnamese(mode: str | None) -> str:
    mapping = {
        "with_g1_g2": "Có điểm G1 và G2",
        "early_prediction": "Dự báo sớm, không có G1 và G2",
    }

    return mapping.get(mode, mode or "Không xác định")


def convert_history_data(records: list[dict]) -> pd.DataFrame:
    """Chuyển dữ liệu API thành DataFrame dùng cho giao diện."""

    rows = []

    for record in records:
        created_at = pd.to_datetime(
            record.get("created_at"),
            errors="coerce",
        )

        formatted_time = (
            created_at.strftime("%d/%m/%Y %H:%M:%S")
            if not pd.isna(created_at)
            else ""
        )

        rows.append(
            {
                "ID": record.get("prediction_id"),
                "Họ tên": record.get("full_name") or "",
                "Môn học": subject_to_vietnamese(
                    record.get("subject")
                ),
                "Chế độ dự đoán": mode_to_vietnamese(
                    record.get("prediction_mode")
                ),
                "Điểm G3 dự đoán": record.get("predicted_g3"),
                "Kết quả": record.get("predicted_result") or "",
                "Model": record.get("model_name") or "",
                "Scenario": record.get("scenario") or "",
                "Phiên bản model": record.get("model_version") or "",
                "Thời gian": formatted_time,
            }
        )

    columns = [
        "ID",
        "Họ tên",
        "Môn học",
        "Chế độ dự đoán",
        "Điểm G3 dự đoán",
        "Kết quả",
        "Model",
        "Scenario",
        "Phiên bản model",
        "Thời gian",
    ]

    return pd.DataFrame(rows, columns=columns)


refresh_col, status_col = st.columns([1, 4])

with refresh_col:
    if st.button(
    "🔄 Làm mới dữ liệu",
    use_container_width=True,
):
        st.rerun()


try:
    history_records = load_prediction_history()
    history_data = convert_history_data(history_records)

    with status_col:
        st.success(
            f"Đã tải {len(history_data)} bản ghi từ SQL Server."
        )

except requests.exceptions.ConnectionError:
    st.error(
        "Không kết nối được FastAPI. "
        "Hãy kiểm tra API đang chạy tại http://127.0.0.1:8000."
    )
    st.stop()

except requests.exceptions.Timeout:
    st.error("FastAPI phản hồi quá lâu.")
    st.stop()

except requests.exceptions.HTTPError as exc:
    st.error(f"FastAPI trả về lỗi: {exc}")
    st.stop()

except Exception as exc:
    st.error(f"Không thể tải lịch sử dự đoán: {exc}")
    st.stop()


st.markdown(
    """
    <div class="section-title">
        <h3>🔎 Bộ lọc tìm kiếm</h3>
        <p>Tìm theo họ tên, môn học, chế độ hoặc khoảng điểm G3.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

filter_col1, filter_col2, filter_col3 = st.columns(
    [2, 1, 1],
    gap="medium",
)

with filter_col1:
    search_name = st.text_input(
        "Họ tên học sinh",
        placeholder="Nhập tên cần tìm",
    )

with filter_col2:
    subject_filter = st.selectbox(
        "Môn học",
        options=[
            "Tất cả",
            "Toán",
            "Tiếng Bồ Đào Nha",
        ],
    )

with filter_col3:
    mode_filter = st.selectbox(
        "Chế độ",
        options=[
            "Tất cả",
            "Có điểm G1 và G2",
            "Dự báo sớm, không có G1 và G2",
        ],
    )

score_range = st.slider(
    "Khoảng điểm G3 dự đoán",
    min_value=0.0,
    max_value=20.0,
    value=(0.0, 20.0),
    step=0.5,
)

filtered_history = history_data.copy()

if search_name.strip():
    filtered_history = filtered_history[
        filtered_history["Họ tên"]
        .astype(str)
        .str.contains(
            search_name.strip(),
            case=False,
            na=False,
        )
    ]

if subject_filter != "Tất cả":
    filtered_history = filtered_history[
        filtered_history["Môn học"] == subject_filter
    ]

if mode_filter != "Tất cả":
    filtered_history = filtered_history[
        filtered_history["Chế độ dự đoán"] == mode_filter
    ]

if not filtered_history.empty:
    filtered_history["Điểm G3 dự đoán"] = pd.to_numeric(
        filtered_history["Điểm G3 dự đoán"],
        errors="coerce",
    )

    filtered_history = filtered_history[
        filtered_history["Điểm G3 dự đoán"].between(
            score_range[0],
            score_range[1],
            inclusive="both",
        )
    ]


st.divider()

st.markdown(
    """
    <div class="section-title">
        <h3>📊 Thống kê lịch sử</h3>
        <p>Tổng hợp các chỉ số từ dữ liệu đang được lọc.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

scores = (
    filtered_history["Điểm G3 dự đoán"].dropna()
    if not filtered_history.empty
    else pd.Series(dtype="float64")
)

total_records = len(filtered_history)
average_score = float(scores.mean()) if not scores.empty else None
minimum_score = float(scores.min()) if not scores.empty else None
maximum_score = float(scores.max()) if not scores.empty else None

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Số bản ghi",
    total_records,
)

metric_col2.metric(
    "G3 trung bình",
    f"{average_score:.2f}"
    if average_score is not None
    else "--",
)

metric_col3.metric(
    "G3 thấp nhất",
    f"{minimum_score:.2f}"
    if minimum_score is not None
    else "--",
)

metric_col4.metric(
    "G3 cao nhất",
    f"{maximum_score:.2f}"
    if maximum_score is not None
    else "--",
)


st.divider()

st.markdown(
    """
    <div class="section-title">
        <h3>🗂️ Danh sách dự đoán</h3>
        <p>Các bản ghi được đọc trực tiếp từ SQL Server.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.dataframe(
    filtered_history,
    use_container_width=True,
    hide_index=True,
)

if filtered_history.empty:
    st.info("Không có dữ liệu phù hợp với bộ lọc hiện tại.")


st.divider()

st.markdown(
    """
    <div class="section-title">
        <h3>📥 Xuất dữ liệu</h3>
        <p>Tải danh sách đang hiển thị dưới dạng CSV.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_history.empty:
    st.button(
        "Tải lịch sử CSV",
        use_container_width=True,
        disabled=True,
    )
else:
    csv_data = filtered_history.to_csv(
        index=False,
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Tải lịch sử CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "Dữ liệu được lưu lâu dài trong SQL Server "
    "và được Streamlit truy xuất thông qua FastAPI."
)