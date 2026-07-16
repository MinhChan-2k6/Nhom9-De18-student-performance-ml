import pandas as pd
import streamlit as st

from streamlit_app.components.styles import load_css


# --------------------------------------------------
# Cấu hình trang
# --------------------------------------------------
st.set_page_config(
    page_title="Dự đoán hàng loạt",
    page_icon="📁",
    layout="wide"
)

load_css()


# --------------------------------------------------
# Tiêu đề trang
# --------------------------------------------------
st.title("📁 Dự đoán hàng loạt")

st.caption(
    "Tải file CSV để kiểm tra và dự đoán điểm G3 "
    "cho nhiều học sinh cùng lúc."
)


# --------------------------------------------------
# Chọn chế độ dự đoán
# --------------------------------------------------
prediction_mode = st.radio(
    "Chế độ dự đoán",
    options=[
        "Có điểm G1 và G2",
        "Dự báo sớm, không có G1 và G2"
    ],
    horizontal=True
)

if prediction_mode == "Có điểm G1 và G2":
    st.markdown(
        """
        <div class="recommend-note">
            <strong>✅ Chế độ khuyến nghị</strong><br>
            File cần có thêm hai cột G1 và G2.
            Chế độ này thường cho kết quả dự đoán G3 chính xác hơn.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="advanced-note">
            <strong>🔎 Chế độ dự báo sớm</strong><br>
            File không cần G1 và G2. Hệ thống dự đoán dựa trên
            các yếu tố cá nhân, gia đình và học tập.
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Hướng dẫn chuẩn bị file
# --------------------------------------------------
with st.expander(
    "📘 Hướng dẫn chuẩn bị file CSV",
    expanded=False
):
    st.write(
        "Mỗi dòng trong file tương ứng với một học sinh."
    )

    st.write(
        "Tên cột phải khớp với các đặc trưng mà model sử dụng."
    )

    st.markdown(
        """
        **Lưu ý:**

        - File phải có định dạng `.csv`.
        - Nên lưu file bằng mã hóa UTF-8.
        - Không cần cột `G3` vì đây là kết quả cần dự đoán.
        - Nếu chọn chế độ có G1 và G2, file phải có đủ hai cột này.
        """
    )


st.divider()


# --------------------------------------------------
# Khu vực tải file
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">
        <h3>📤 Tải dữ liệu đầu vào</h3>
        <p>Chọn file CSV cần kiểm tra và dự đoán hàng loạt.</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Chọn file CSV",
    type=["csv"],
    help="Chỉ hỗ trợ file có phần mở rộng .csv"
)


# --------------------------------------------------
# Trạng thái khi chưa tải file
# --------------------------------------------------
if uploaded_file is None:
    st.info(
        "Vui lòng tải lên một file CSV để bắt đầu kiểm tra dữ liệu."
    )

    st.stop()


# --------------------------------------------------
# Đọc file CSV
# --------------------------------------------------
try:
    batch_data = pd.read_csv(uploaded_file)

except pd.errors.EmptyDataError:
    st.error("File CSV không chứa dữ liệu.")
    st.stop()

except pd.errors.ParserError:
    st.error(
        "Không thể đọc cấu trúc file CSV. "
        "Vui lòng kiểm tra dấu phân cách và định dạng file."
    )
    st.stop()

except UnicodeDecodeError:
    st.error(
        "Không thể đọc mã hóa của file. "
        "Hãy lưu lại file dưới định dạng UTF-8."
    )
    st.stop()

except Exception as error:
    st.error(f"Đã xảy ra lỗi khi đọc file: {error}")
    st.stop()


# --------------------------------------------------
# Kiểm tra dữ liệu cơ bản
# --------------------------------------------------
if batch_data.empty:
    st.error("File CSV không có dòng dữ liệu nào.")
    st.stop()


duplicate_columns = (
    batch_data.columns[
        batch_data.columns.duplicated()
    ]
    .tolist()
)

if duplicate_columns:
    st.error(
        "File có tên cột bị trùng: "
        + ", ".join(duplicate_columns)
    )
    st.stop()


# --------------------------------------------------
# Thông tin tổng quan
# --------------------------------------------------
st.success(
    f"Đã đọc file thành công: "
    f"{batch_data.shape[0]} dòng và "
    f"{batch_data.shape[1]} cột."
)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric(
        label="Số học sinh",
        value=batch_data.shape[0]
    )

with summary_col2:
    st.metric(
        label="Số cột",
        value=batch_data.shape[1]
    )

with summary_col3:
    missing_value_count = int(
        batch_data.isna().sum().sum()
    )

    st.metric(
        label="Giá trị bị thiếu",
        value=missing_value_count
    )


# --------------------------------------------------
# Xem trước dữ liệu
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">
        <h3>👀 Xem trước dữ liệu</h3>
        <p>Hiển thị tối đa 10 dòng đầu tiên của file.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.dataframe(
    batch_data.head(10),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Danh sách cột bắt buộc
# --------------------------------------------------
required_columns = [
    "subject",
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"
]

if prediction_mode == "Có điểm G1 và G2":
    required_columns.extend(["G1", "G2"])


missing_columns = [
    column
    for column in required_columns
    if column not in batch_data.columns
]


# --------------------------------------------------
# Kiểm tra cấu trúc file
# --------------------------------------------------
st.markdown(
    """
    <div class="section-title">
        <h3>✅ Kiểm tra dữ liệu</h3>
        <p>Hệ thống kiểm tra cột, dữ liệu thiếu và miền giá trị.</p>
    </div>
    """,
    unsafe_allow_html=True
)

file_is_valid = True


if "G3" in batch_data.columns:
    st.warning(
        "File đang có cột G3. "
        "Cột này là nhãn cần dự đoán và sẽ không được gửi vào model."
    )


if missing_columns:
    st.error(
        "File đang thiếu các cột bắt buộc: "
        + ", ".join(missing_columns)
    )

    file_is_valid = False

else:
    st.success(
        "File có đầy đủ các cột bắt buộc cho chế độ đã chọn."
    )


# --------------------------------------------------
# Kiểm tra dữ liệu thiếu
# --------------------------------------------------
if file_is_valid:
    missing_required_values = (
        batch_data[required_columns]
        .isna()
        .sum()
    )

    columns_with_missing_values = (
        missing_required_values[
            missing_required_values > 0
        ]
        .index
        .tolist()
    )

    if columns_with_missing_values:
        st.error(
            "Các cột bắt buộc đang có dữ liệu trống: "
            + ", ".join(columns_with_missing_values)
        )

        file_is_valid = False


# --------------------------------------------------
# Kiểm tra miền giá trị
# --------------------------------------------------
validation_errors = []

if file_is_valid:
    if not batch_data["age"].between(15, 22).all():
        validation_errors.append(
            "Cột age phải nằm trong khoảng từ 15 đến 22."
        )

    if not batch_data["studytime"].between(1, 4).all():
        validation_errors.append(
            "Cột studytime phải nằm trong khoảng từ 1 đến 4."
        )

    if not batch_data["failures"].between(0, 3).all():
        validation_errors.append(
            "Cột failures phải nằm trong khoảng từ 0 đến 3."
        )

    if not batch_data["absences"].between(0, 75).all():
        validation_errors.append(
            "Cột absences phải nằm trong khoảng từ 0 đến 75."
        )

    if prediction_mode == "Có điểm G1 và G2":
        if not batch_data["G1"].between(0, 20).all():
            validation_errors.append(
                "Cột G1 phải nằm trong khoảng từ 0 đến 20."
            )

        if not batch_data["G2"].between(0, 20).all():
            validation_errors.append(
                "Cột G2 phải nằm trong khoảng từ 0 đến 20."
            )


if validation_errors:
    for validation_error in validation_errors:
        st.error(validation_error)

    file_is_valid = False


# --------------------------------------------------
# Trạng thái cuối cùng
# --------------------------------------------------
if file_is_valid:
    st.markdown(
        """
        <div class="recommend-note">
            <strong>✅ File hợp lệ</strong><br>
            Dữ liệu đã vượt qua các bước kiểm tra cơ bản
            và sẵn sàng gửi đến model.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning(
        "Hãy sửa các lỗi trong file trước khi thực hiện dự đoán."
    )


# --------------------------------------------------
# Nút dự đoán hàng loạt
# --------------------------------------------------
predict_batch = st.button(
    "🤖 Dự đoán toàn bộ file",
    use_container_width=True,
    disabled=not file_is_valid
)


# --------------------------------------------------
# Xử lý khi nhấn nút
# --------------------------------------------------
if predict_batch:
    st.success(
        "File đã vượt qua bước kiểm tra dữ liệu."
    )

    st.info(
        "Model và API chưa được kết nối. "
        "Khi endpoint POST /predict/batch hoàn thành, "
        "dữ liệu sẽ được gửi đến model và file kết quả thật "
        "sẽ được tạo để tải xuống."
    )

    request_summary = {
        "prediction_mode": (
            "with_g1_g2"
            if prediction_mode == "Có điểm G1 và G2"
            else "early_prediction"
        ),
        "total_rows": int(batch_data.shape[0]),
        "total_columns": int(batch_data.shape[1]),
        "required_columns": required_columns
    }

    with st.expander(
        "🛠 Dữ liệu kỹ thuật dành cho nhóm phát triển"
    ):
        st.json(request_summary)