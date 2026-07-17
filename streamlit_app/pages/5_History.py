import pandas as pd
import streamlit as st

from streamlit_app.components.styles import load_css


st.set_page_config(
    page_title="Lịch sử dự đoán",
    page_icon="📜",
    layout="wide",
)

load_css()

st.title("📜 Lịch sử dự đoán")
st.caption("Tra cứu các lần dự đoán điểm G3 đã thực hiện trong phiên hiện tại.")

history_records = st.session_state.get("prediction_history", [])
history_data = pd.DataFrame(history_records)

expected_columns = [
    "Họ tên",
    "Môn học",
    "Chế độ dự đoán",
    "Điểm G3 dự đoán",
    "Model",
    "Scenario",
    "Thời gian",
]
for column in expected_columns:
    if column not in history_data.columns:
        history_data[column] = pd.Series(dtype="object")

if not history_data.empty:
    history_data.insert(0, "ID", range(1, len(history_data) + 1))
else:
    history_data = pd.DataFrame(columns=["ID", *expected_columns])

st.markdown(
    """
    <div class="section-title">
        <h3>🔎 Bộ lọc tìm kiếm</h3>
        <p>Tìm theo họ tên, môn học, chế độ dự đoán hoặc khoảng điểm G3.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1], gap="medium")

with filter_col1:
    search_name = st.text_input(
        "Họ tên học sinh",
        placeholder="Nhập tên cần tìm",
    )

with filter_col2:
    subject_filter = st.selectbox(
        "Môn học",
        options=["Tất cả", "Toán", "Tiếng Bồ Đào Nha"],
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
        .str.contains(search_name.strip(), case=False, na=False)
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
        filtered_history["Điểm G3 dự đoán"], errors="coerce"
    )
    filtered_history = filtered_history[
        filtered_history["Điểm G3 dự đoán"].between(
            score_range[0], score_range[1], inclusive="both"
        )
    ]

st.divider()

st.markdown(
    """
    <div class="section-title">
        <h3>📊 Thống kê lịch sử</h3>
        <p>Tổng hợp số bản ghi và các thống kê mô tả của điểm G3.</p>
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

with metric_col1:
    st.metric("Số bản ghi", total_records)

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

st.markdown(
    """
    <div class="section-title">
        <h3>🗂️ Danh sách dự đoán</h3>
        <p>Hiển thị các bản ghi phù hợp với điều kiện tìm kiếm.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.dataframe(filtered_history, use_container_width=True, hide_index=True)

if filtered_history.empty:
    st.info("Chưa có dữ liệu lịch sử phù hợp với bộ lọc hiện tại.")

st.divider()

st.markdown(
    """
    <div class="section-title">
        <h3>📥 Xuất dữ liệu</h3>
        <p>Tải danh sách lịch sử đang hiển thị dưới dạng CSV.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_history.empty:
    st.button("Tải lịch sử CSV", use_container_width=True, disabled=True)
    st.caption("Chưa có dữ liệu để xuất file.")
else:
    csv_data = filtered_history.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Tải lịch sử CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "Lịch sử hiện được lưu trong session của Streamlit; khi kết nối SQL Server, "
    "dữ liệu sẽ được lưu lâu dài."
)
