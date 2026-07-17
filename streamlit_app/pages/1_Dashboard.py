import pandas as pd
import streamlit as st

from streamlit_app.components.styles import load_css


# --------------------------------------------------
# Cấu hình trang
# --------------------------------------------------
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

load_css()


# --------------------------------------------------
# Tiêu đề trang
# --------------------------------------------------
st.title("📊 Dashboard")

st.caption(
    "Tổng quan hoạt động của hệ thống dự đoán kết quả học tập."
)


# --------------------------------------------------
# Dữ liệu tạm thời
# Sau này sẽ lấy từ SQL Server và API
# --------------------------------------------------
total_predictions = 0
average_predicted_g3 = None
pass_rate = None
active_model = None

pass_count = 0
fail_count = 0

recent_score_data = pd.DataFrame(
    columns=[
        "Lần dự đoán",
        "Điểm G3"
    ]
)

recent_predictions = pd.DataFrame(
    columns=[
        "Họ tên",
        "Chế độ dự đoán",
        "Điểm G3",
        "Kết quả quy đổi",
        "Model",
        "Thời gian"
    ]
)


# --------------------------------------------------
# Lời chào và trạng thái tổng quan
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">
        <h3>🎓 Tổng quan hệ thống</h3>
        <p>
            Theo dõi số lượt dự đoán, điểm G3 trung bình,
            kết quả quy đổi và trạng thái các thành phần hệ thống.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Các thẻ thống kê
# --------------------------------------------------
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        label="Tổng dự đoán",
        value=total_predictions
    )

with metric_col2:
    average_score_display = (
        f"{average_predicted_g3:.2f}"
        if average_predicted_g3 is not None
        else "--"
    )

    st.metric(
        label="G3 trung bình",
        value=average_score_display
    )

with metric_col3:
    pass_rate_display = (
        f"{pass_rate:.1f}%"
        if pass_rate is not None
        else "--"
    )

    st.metric(
        label="Tỷ lệ đạt",
        value=pass_rate_display
    )

with metric_col4:
    st.metric(
        label="Model",
        value=active_model or "Chưa tải"
    )


st.divider()


# --------------------------------------------------
# Khu vực biểu đồ
# --------------------------------------------------
chart_col1, chart_col2 = st.columns(
    [1, 1],
    gap="large"
)

with chart_col1:
    st.markdown(
        """
        <div class="section-title">
            <h3>📈 Phân bố kết quả quy đổi</h3>
            <p>
                Đạt hoặc trượt được suy ra từ điểm G3 dự đoán.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if pass_count == 0 and fail_count == 0:
        st.info(
            "Chưa có dữ liệu kết quả để hiển thị biểu đồ."
        )

    else:
        result_summary = pd.DataFrame(
            {
                "Kết quả": [
                    "Đạt",
                    "Trượt"
                ],
                "Số lượng": [
                    pass_count,
                    fail_count
                ]
            }
        )

        st.bar_chart(
            result_summary,
            x="Kết quả",
            y="Số lượng"
        )


with chart_col2:
    st.markdown(
        """
        <div class="section-title">
            <h3>📉 Điểm dự đoán gần đây</h3>
            <p>
                Theo dõi xu hướng điểm G3 của các lần dự đoán mới nhất.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if recent_score_data.empty:
        st.info(
            "Chưa có điểm dự đoán gần đây."
        )

    else:
        st.line_chart(
            recent_score_data,
            x="Lần dự đoán",
            y="Điểm G3"
        )


st.divider()


# --------------------------------------------------
# Trạng thái hệ thống
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">
        <h3>⚙️ Trạng thái hệ thống</h3>
        <p>
            Kiểm tra nhanh trạng thái API, cơ sở dữ liệu và model.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.info(
        "🌐 API\n\nChưa kết nối"
    )

with status_col2:
    st.info(
        "🗄️ SQL Server\n\nChưa kết nối"
    )

with status_col3:
    st.info(
        "🤖 Model\n\nChưa tải"
    )


st.divider()


# --------------------------------------------------
# Bảng dự đoán gần đây
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">
        <h3>🕘 Dự đoán gần đây</h3>
        <p>
            Hiển thị các lần dự đoán mới nhất được lưu trong hệ thống.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.dataframe(
    recent_predictions,
    use_container_width=True,
    hide_index=True
)

if recent_predictions.empty:
    st.warning(
        "Chưa có dữ liệu dự đoán. "
        "Hãy thực hiện dự đoán đầu tiên để cập nhật Dashboard."
    )


# --------------------------------------------------
# Ghi chú nghiệp vụ
# --------------------------------------------------
st.caption(
    "Lưu ý: Đạt/Trượt là kết quả quy đổi từ điểm G3 dự đoán. "
    "Mô hình chính vẫn là mô hình hồi quy."
)