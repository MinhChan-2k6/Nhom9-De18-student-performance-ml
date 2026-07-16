import pandas as pd
import streamlit as st

from streamlit_app.components.styles import load_css


# --------------------------------------------------
# Cấu hình trang
# --------------------------------------------------
st.set_page_config(
    page_title="Lịch sử dự đoán",
    page_icon="📜",
    layout="wide"
)

load_css()


# --------------------------------------------------
# Tiêu đề trang
# --------------------------------------------------
st.title("📜 Lịch sử dự đoán")

st.caption(
    "Tra cứu và theo dõi các lần dự đoán đã được lưu trong hệ thống."
)


# --------------------------------------------------
# Dữ liệu tạm thời
# Sau này sẽ lấy từ SQL Server
# --------------------------------------------------
history_data = pd.DataFrame(
    columns=[
        "ID",
        "Họ tên",
        "Môn học",
        "Chế độ dự đoán",
        "Điểm G3 dự đoán",
        "Kết quả quy đổi",
        "Model",
        "Thời gian"
    ]
)


# --------------------------------------------------
# Khu vực bộ lọc
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">
        <h3>🔎 Bộ lọc tìm kiếm</h3>
        <p>
            Tìm kiếm lịch sử theo họ tên, môn học,
            chế độ dự đoán hoặc kết quả quy đổi.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(
    [2, 1, 1, 1],
    gap="medium"
)

with filter_col1:
    search_name = st.text_input(
        "Họ tên học sinh",
        placeholder="Nhập tên cần tìm"
    )

with filter_col2:
    subject_filter = st.selectbox(
        "Môn học",
        options=[
            "Tất cả",
            "Toán",
            "Tiếng Bồ Đào Nha"
        ]
    )

with filter_col3:
    mode_filter = st.selectbox(
        "Chế độ",
        options=[
            "Tất cả",
            "Có G1 và G2",
            "Dự báo sớm"
        ]
    )

with filter_col4:
    result_filter = st.selectbox(
        "Kết quả quy đổi",
        options=[
            "Tất cả",
            "Đạt",
            "Trượt"
        ]
    )


# --------------------------------------------------
# Áp dụng bộ lọc
# --------------------------------------------------
filtered_history = history_data.copy()

if search_name.strip():
    filtered_history = filtered_history[
        filtered_history["Họ tên"]
        .astype(str)
        .str.contains(
            search_name.strip(),
            case=False,
            na=False
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

if result_filter != "Tất cả":
    filtered_history = filtered_history[
        filtered_history["Kết quả quy đổi"] == result_filter
    ]


st.divider()


# --------------------------------------------------
# Thống kê nhanh
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">
        <h3>📊 Thống kê lịch sử</h3>
        <p>
            Tổng hợp nhanh số lượng bản ghi và điểm dự đoán.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

total_records = len(filtered_history)

average_score = (
    filtered_history["Điểm G3 dự đoán"].mean()
    if not filtered_history.empty
    else None
)

pass_count = (
    int(
        (
            filtered_history["Kết quả quy đổi"] == "Đạt"
        ).sum()
    )
    if not filtered_history.empty
    else 0
)

fail_count = (
    int(
        (
            filtered_history["Kết quả quy đổi"] == "Trượt"
        ).sum()
    )
    if not filtered_history.empty
    else 0
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        label="Số bản ghi",
        value=total_records
    )

with metric_col2:
    st.metric(
        label="G3 trung bình",
        value=(
            f"{average_score:.2f}"
            if average_score is not None
            else "--"
        )
    )

with metric_col3:
    st.metric(
        label="Số kết quả đạt",
        value=pass_count
    )

with metric_col4:
    st.metric(
        label="Số kết quả trượt",
        value=fail_count
    )


st.divider()


# --------------------------------------------------
# Bảng lịch sử
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">
        <h3>🗂️ Danh sách dự đoán</h3>
        <p>
            Hiển thị các bản ghi phù hợp với điều kiện tìm kiếm.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.dataframe(
    filtered_history,
    use_container_width=True,
    hide_index=True
)

if filtered_history.empty:
    st.info(
        "Chưa có dữ liệu lịch sử phù hợp với bộ lọc hiện tại."
    )


# --------------------------------------------------
# Chức năng xuất dữ liệu
# --------------------------------------------------
st.divider()

st.markdown(
    """
    <div class="section-title">
        <h3>📥 Xuất dữ liệu</h3>
        <p>
            Tải danh sách lịch sử đang hiển thị dưới dạng CSV.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if filtered_history.empty:
    st.button(
        "Tải lịch sử CSV",
        use_container_width=True,
        disabled=True
    )

    st.caption(
        "Chưa có dữ liệu để xuất file."
    )

else:
    csv_data = filtered_history.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Tải lịch sử CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )


# --------------------------------------------------
# Ghi chú hệ thống
# --------------------------------------------------
st.caption(
    "Trang History hiện sử dụng DataFrame rỗng. "
    "Sau khi kết nối SQL Server, dữ liệu sẽ được truy vấn "
    "theo bộ lọc và hiển thị tại đây."
)