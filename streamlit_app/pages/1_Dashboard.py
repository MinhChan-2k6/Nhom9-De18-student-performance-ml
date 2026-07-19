from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

from streamlit_app.components.styles import (
    apply_dark_mode,
    load_css,
)
from streamlit_app.config import API_BASE_URL



st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
)

load_css()


# ==================================================
# CSS RIÊNG CHO DASHBOARD
# ==================================================
st.markdown(
    """
    <style>
    .dashboard-hero {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #155EEF 0%,
            #5B4BDB 55%,
            #7A5AF8 100%
        );
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 14px 32px rgba(21, 94, 239, 0.18);
    }

    .dashboard-hero h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }

    .dashboard-hero p {
        margin: 0.6rem 0 0;
        color: rgba(255, 255, 255, 0.88);
    }

    .status-card {
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid #E4E7EC;
        background: white;
        min-height: 125px;
    }

    .status-title {
        color: #667085;
        font-size: 0.85rem;
        font-weight: 700;
    }

    .status-value {
        color: #101828;
        font-size: 1.15rem;
        font-weight: 800;
        margin-top: 0.4rem;
    }

    .status-online {
        color: #027A48;
    }

    .status-offline {
        color: #B42318;
    }

    .status-note {
        color: #98A2B3;
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
apply_dark_mode()

# ==================================================
# HÀM GỌI API
# ==================================================
@st.cache_data(ttl=15, show_spinner=False)
def load_dashboard_data() -> dict[str, Any]:
    result = {
        "api_online": False,
        "database_online": False,
        "health": {},
        "history": [],
        "models": [],
    }

    # Kiểm tra API
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )
        response.raise_for_status()

        result["api_online"] = True
        result["health"] = response.json()

    except requests.RequestException:
        pass

    # Kiểm tra History + Database
    try:
        response = requests.get(
            f"{API_BASE_URL}/history",
            params={"limit": 500},
            timeout=10,
        )
        response.raise_for_status()

        history = response.json()

        if isinstance(history, list):
            result["history"] = history
            result["database_online"] = True

    except requests.RequestException:
        pass

    # Lấy thông tin model
    try:
        response = requests.get(
            f"{API_BASE_URL}/model-info",
            timeout=10,
        )
        response.raise_for_status()

        models = response.json()

        if isinstance(models, list):
            result["models"] = models

    except requests.RequestException:
        pass

    return result


def subject_label(value: str | None) -> str:
    mapping = {
        "math": "Toán",
        "portuguese": "Tiếng Bồ Đào Nha",
    }

    return mapping.get(
        value,
        value or "Không xác định",
    )


def mode_label(value: str | None) -> str:
    mapping = {
        "with_g1_g2": "Có G1 và G2",
        "early_prediction": "Dự báo sớm",
    }

    return mapping.get(
        value,
        value or "Không xác định",
    )


def convert_history(
    records: list[dict],
) -> pd.DataFrame:
    """
    Chuyển dữ liệu lịch sử từ API thành DataFrame.

    Bài toán thuần hồi quy nên chỉ sử dụng điểm G3 dự đoán,
    không có kết quả Đạt/Trượt.
    """

    rows = []

    for record in records:
        rows.append(
            {
                "ID": record.get("prediction_id"),
                "Họ tên": record.get("full_name") or "",
                "Môn học": subject_label(
                    record.get("subject")
                ),
                "Chế độ": mode_label(
                    record.get("prediction_mode")
                ),
                "Điểm G3": record.get("predicted_g3"),
                "Model": record.get("model_name"),
                "Scenario": record.get("scenario"),
                "Thời gian": record.get("created_at"),
            }
        )

    return pd.DataFrame(rows)


# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <div class="dashboard-hero">
        <h1>📊 Dashboard</h1>
        <p>
            Tổng quan hoạt động của hệ thống hồi quy dự đoán điểm G3,
            dữ liệu lịch sử và trạng thái các thành phần.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# NÚT REFRESH
# ==================================================
refresh_col, note_col = st.columns([1, 4])

with refresh_col:
    if st.button(
        "🔄 Làm mới dữ liệu",
        use_container_width=True,
    ):
        load_dashboard_data.clear()
        st.rerun()

with note_col:
    st.caption(
        "Dashboard lấy dữ liệu trực tiếp từ FastAPI và SQL Server."
    )


# ==================================================
# LOAD DATA
# ==================================================
dashboard_data = load_dashboard_data()

api_online = dashboard_data["api_online"]
database_online = dashboard_data["database_online"]
history_records = dashboard_data["history"]
model_records = dashboard_data["models"]

history_df = convert_history(history_records)


# ==================================================
# TÍNH TOÁN THỐNG KÊ HỒI QUY
# ==================================================
total_predictions = len(history_df)

average_score = None
minimum_score = None
maximum_score = None

if not history_df.empty:
    scores = pd.to_numeric(
        history_df["Điểm G3"],
        errors="coerce",
    ).dropna()

    if not scores.empty:
        average_score = float(scores.mean())
        minimum_score = float(scores.min())
        maximum_score = float(scores.max())


# ==================================================
# TỔNG QUAN
# ==================================================
st.markdown(
    """
    <div class="section-title">
        <h3>🎓 Tổng quan hệ thống</h3>
        <p>
            Các thống kê hồi quy được tính từ lịch sử dự đoán
            đang lưu trong SQL Server.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Tổng dự đoán",
    total_predictions,
)

metric_col2.metric(
    "G3 trung bình",
    (
        f"{average_score:.2f}"
        if average_score is not None
        else "--"
    ),
)

metric_col3.metric(
    "G3 thấp nhất",
    (
        f"{minimum_score:.2f}"
        if minimum_score is not None
        else "--"
    ),
)

metric_col4.metric(
    "G3 cao nhất",
    (
        f"{maximum_score:.2f}"
        if maximum_score is not None
        else "--"
    ),
)


st.divider()


# ==================================================
# BIỂU ĐỒ
# ==================================================
chart_col1, chart_col2 = st.columns(
    [1, 1],
    gap="large",
)


# --------------------------------------------------
# PHÂN BỐ ĐIỂM G3
# --------------------------------------------------
with chart_col1:

    st.markdown(
        """
        <div class="section-title">
            <h3>📈 Phân bố điểm G3</h3>
            <p>
                Số lượt dự đoán theo từng khoảng điểm.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history_df.empty:

        st.info(
            "Chưa có dữ liệu để hiển thị biểu đồ."
        )

    else:

        scores = pd.to_numeric(
            history_df["Điểm G3"],
            errors="coerce",
        ).dropna()

        if scores.empty:

            st.info(
                "Không có dữ liệu điểm hợp lệ."
            )

        else:

            intervals = pd.cut(
                scores,
                bins=[
                    0,
                    4,
                    8,
                    12,
                    16,
                    20,
                ],
                include_lowest=True,
                labels=[
                    "0–4",
                    ">4–8",
                    ">8–12",
                    ">12–16",
                    ">16–20",
                ],
            )

            distribution = (
                intervals
                .value_counts(sort=False)
                .rename_axis("Khoảng điểm")
                .reset_index(name="Số lượng")
            )

            # Biểu đồ Altair để nhãn trục X nằm ngang
            chart = (
                alt.Chart(distribution)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Khoảng điểm:N",
                        title="Khoảng điểm",
                        sort=None,
                        axis=alt.Axis(
                            labelAngle=0,
                            labelOverlap=False,
                            labelPadding=8,
                        ),
                    ),
                    y=alt.Y(
                        "Số lượng:Q",
                        title="Số lượng",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Khoảng điểm:N",
                            title="Khoảng điểm",
                        ),
                        alt.Tooltip(
                            "Số lượng:Q",
                            title="Số lượng",
                        ),
                    ],
                )
                .properties(
                    height=300,
                )
            )

            st.altair_chart(
                chart,
                use_container_width=True,
            )


# --------------------------------------------------
# ĐIỂM DỰ ĐOÁN GẦN ĐÂY
# --------------------------------------------------
with chart_col2:

    st.markdown(
        """
        <div class="section-title">
            <h3>📉 Điểm dự đoán gần đây</h3>
            <p>
                Theo dõi 10 giá trị G3 dự đoán mới nhất.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history_df.empty:

        st.info(
            "Chưa có điểm dự đoán gần đây."
        )

    else:

        recent_scores = history_df.head(10).copy()

        recent_scores = recent_scores.iloc[::-1]

        recent_scores["Điểm G3"] = pd.to_numeric(
            recent_scores["Điểm G3"],
            errors="coerce",
        )

        recent_scores = recent_scores.dropna(
            subset=["Điểm G3"]
        )

        recent_scores["Lần dự đoán"] = range(
            1,
            len(recent_scores) + 1,
        )

        if recent_scores.empty:

            st.info(
                "Không có dữ liệu điểm hợp lệ."
            )

        else:

            st.line_chart(
                recent_scores,
                x="Lần dự đoán",
                y="Điểm G3",
            )


st.divider()


# ==================================================
# TRẠNG THÁI HỆ THỐNG
# ==================================================
st.markdown(
    """
    <div class="section-title">
        <h3>⚙️ Trạng thái hệ thống</h3>
        <p>
            Kiểm tra trạng thái API, SQL Server và model hồi quy.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


status_col1, status_col2, status_col3 = st.columns(3)


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------
with status_col1:

    api_class = (
        "status-online"
        if api_online
        else "status-offline"
    )

    api_text = (
        "Đang hoạt động"
        if api_online
        else "Mất kết nối"
    )

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">
                🌐 FastAPI
            </div>
            <div class="status-value {api_class}">
                {api_text}
            </div>
            <div class="status-note">
                {API_BASE_URL}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# SQL SERVER
# --------------------------------------------------
with status_col2:

    db_class = (
        "status-online"
        if database_online
        else "status-offline"
    )

    db_text = (
        "Đã kết nối"
        if database_online
        else "Không kết nối"
    )

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">
                🗄️ SQL Server
            </div>
            <div class="status-value {db_class}">
                {db_text}
            </div>
            <div class="status-note">
                StudentPerformanceDB
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# MODEL
# --------------------------------------------------
with status_col3:

    model_names = []

    for model in model_records:

        model_name = model.get(
            "best_model"
        )

        if model_name:
            model_names.append(
                str(model_name)
            )

    unique_models = list(
        dict.fromkeys(model_names)
    )

    model_text = (
        ", ".join(unique_models)
        if unique_models
        else "Chưa tải"
    )

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">
                🤖 Model hồi quy
            </div>
            <div class="status-value">
                {model_text}
            </div>
            <div class="status-note">
                {len(model_records)} kịch bản đang sử dụng
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ==================================================
# DỰ ĐOÁN GẦN ĐÂY
# ==================================================
st.markdown(
    """
    <div class="section-title">
        <h3>🕘 Dự đoán gần đây</h3>
        <p>
            10 giá trị G3 dự đoán mới nhất được lưu trong SQL Server.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


if history_df.empty:

    st.warning(
        "Chưa có dữ liệu dự đoán. "
        "Hãy thực hiện dự đoán đầu tiên."
    )

else:

    recent_predictions = history_df.head(10).copy()

    st.dataframe(
        recent_predictions,
        use_container_width=True,
        hide_index=True,
    )


st.caption(
    "Dashboard cập nhật dữ liệu thông qua FastAPI; "
    "lịch sử dự đoán hồi quy được lưu lâu dài trong SQL Server."
)