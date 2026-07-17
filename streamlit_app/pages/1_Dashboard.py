import pandas as pd
import streamlit as st

from streamlit_app.components.styles import load_css


st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
)

load_css()

st.title("📊 Dashboard")
st.caption("Tổng quan các lần dự đoán điểm G3 của hệ thống hồi quy.")

history_records = st.session_state.get("prediction_history", [])
history_df = pd.DataFrame(history_records)

score_column = "Điểm G3 dự đoán"
total_predictions = len(history_df)
average_score = None
minimum_score = None
maximum_score = None

if not history_df.empty and score_column in history_df.columns:
    numeric_scores = pd.to_numeric(history_df[score_column], errors="coerce").dropna()
    if not numeric_scores.empty:
        average_score = float(numeric_scores.mean())
        minimum_score = float(numeric_scores.min())
        maximum_score = float(numeric_scores.max())

st.markdown(
    """
    <div class="section-title">
        <h3>🎓 Tổng quan hệ thống</h3>
        <p>
            Theo dõi số lượt dự đoán và các thống kê mô tả của điểm G3.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Tổng dự đoán", total_predictions)

with metric_col2:
    st.metric(
        "G3 trung bình",
        f"{average_score:.2f}" if average_score is not None else "--",
    )

with metric_col3:
    st.metric(
        "G3 thấp nhất",
        f"{minimum_score:.2f}" if minimum_score is not None else "--",
    )

with metric_col4:
    st.metric(
        "G3 cao nhất",
        f"{maximum_score:.2f}" if maximum_score is not None else "--",
    )

st.divider()

chart_col1, chart_col2 = st.columns([1, 1], gap="large")

with chart_col1:
    st.markdown(
        """
        <div class="section-title">
            <h3>📈 Phân bố điểm G3 dự đoán</h3>
            <p>Số lượt dự đoán trong từng khoảng điểm trên thang 0–20.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history_df.empty or score_column not in history_df.columns:
        st.info("Chưa có dữ liệu điểm để hiển thị biểu đồ.")
    else:
        scores = pd.to_numeric(history_df[score_column], errors="coerce").dropna()
        if scores.empty:
            st.info("Chưa có dữ liệu điểm hợp lệ để hiển thị biểu đồ.")
        else:
            intervals = pd.cut(
                scores,
                bins=[0, 4, 8, 12, 16, 20],
                include_lowest=True,
                labels=["0–4", ">4–8", ">8–12", ">12–16", ">16–20"],
            )
            distribution = (
                intervals.value_counts(sort=False)
                .rename_axis("Khoảng điểm")
                .reset_index(name="Số lượng")
            )
            st.bar_chart(distribution, x="Khoảng điểm", y="Số lượng")

with chart_col2:
    st.markdown(
        """
        <div class="section-title">
            <h3>📉 Điểm dự đoán gần đây</h3>
            <p>Theo dõi điểm G3 của tối đa 10 lần dự đoán mới nhất.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history_df.empty or score_column not in history_df.columns:
        st.info("Chưa có điểm dự đoán gần đây.")
    else:
        recent_scores = history_df.tail(10).copy()
        recent_scores["Lần dự đoán"] = range(1, len(recent_scores) + 1)
        recent_scores[score_column] = pd.to_numeric(
            recent_scores[score_column], errors="coerce"
        )
        recent_scores = recent_scores.dropna(subset=[score_column])
        if recent_scores.empty:
            st.info("Chưa có điểm dự đoán hợp lệ.")
        else:
            st.line_chart(
                recent_scores,
                x="Lần dự đoán",
                y=score_column,
            )

st.divider()

st.markdown(
    """
    <div class="section-title">
        <h3>⚙️ Trạng thái hệ thống</h3>
        <p>Trạng thái hiện tại của API, cơ sở dữ liệu và model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.info("🌐 API\n\nKiểm tra khi thực hiện dự đoán")

with status_col2:
    st.info("🗄️ SQL Server\n\nChưa kết nối")

with status_col3:
    active_model = (
        str(history_df.iloc[-1]["Model"])
        if not history_df.empty and "Model" in history_df.columns
        else "Chưa tải"
    )
    st.info(f"🤖 Model\n\n{active_model}")

st.divider()

st.markdown(
    """
    <div class="section-title">
        <h3>🕘 Dự đoán gần đây</h3>
        <p>Hiển thị tối đa 10 kết quả dự đoán G3 mới nhất.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

recent_predictions = history_df.tail(10).iloc[::-1] if not history_df.empty else history_df
st.dataframe(recent_predictions, use_container_width=True, hide_index=True)

if recent_predictions.empty:
    st.warning(
        "Chưa có dữ liệu dự đoán. Hãy thực hiện dự đoán đầu tiên để cập nhật Dashboard."
    )

st.caption("Đầu ra của hệ thống là điểm G3 dạng số trên thang 0–20.")
