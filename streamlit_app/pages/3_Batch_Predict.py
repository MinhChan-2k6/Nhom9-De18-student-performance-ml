"""Trang Streamlit dự đoán điểm G3 hàng loạt từ file CSV."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from streamlit_app.components.styles import load_css
from streamlit_app.config import API_BASE_URL


st.set_page_config(
    page_title="Dự đoán hàng loạt",
    page_icon="📁",
    layout="wide",
)
load_css()

API_URL = f"{API_BASE_URL}/predict/batch"
REQUEST_TIMEOUT_SECONDS = 120

MODE_LABEL_TO_VALUE = {
    "Có điểm G1 và G2": "with_g1_g2",
    "Dự báo sớm, không có G1 và G2": "early_prediction",
}

BASE_REQUIRED_COLUMNS = [
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
    "absences",
]

NUMERIC_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 3),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 75),
    "G1": (0, 20),
    "G2": (0, 20),
}

CATEGORICAL_VALUES = {
    "subject": {"math", "portuguese"},
    "school": {"gp", "ms"},
    "sex": {"f", "m"},
    "address": {"u", "r"},
    "famsize": {"le3", "gt3"},
    "Pstatus": {"t", "a"},
    "Mjob": {"teacher", "health", "services", "at_home", "other"},
    "Fjob": {"teacher", "health", "services", "at_home", "other"},
    "reason": {"course", "home", "reputation", "other"},
    "guardian": {"mother", "father", "other"},
    "schoolsup": {"yes", "no"},
    "famsup": {"yes", "no"},
    "paid": {"yes", "no"},
    "activities": {"yes", "no"},
    "nursery": {"yes", "no"},
    "higher": {"yes", "no"},
    "internet": {"yes", "no"},
    "romantic": {"yes", "no"},
}

OUTPUT_COLUMNS = {"predicted_g3", "model", "scenario"}


def get_required_columns(prediction_mode: str) -> list[str]:
    """Trả về danh sách cột bắt buộc cho chế độ được chọn."""

    required_columns = list(BASE_REQUIRED_COLUMNS)
    if prediction_mode == "with_g1_g2":
        required_columns.extend(["G1", "G2"])
    return required_columns


def validate_uploaded_dataframe(
    dataframe: pd.DataFrame,
    prediction_mode: str,
) -> tuple[list[str], list[str]]:
    """Kiểm tra nhanh CSV trước khi gửi đến FastAPI."""

    errors: list[str] = []
    warnings: list[str] = []
    required_columns = get_required_columns(prediction_mode)

    duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicate_columns:
        errors.append(
            "Tên cột bị trùng: " + ", ".join(map(str, duplicate_columns))
        )

    conflicting_columns = sorted(OUTPUT_COLUMNS.intersection(dataframe.columns))
    if conflicting_columns:
        errors.append(
            "Hãy xóa các cột kết quả cũ trước khi dự đoán: "
            + ", ".join(conflicting_columns)
        )

    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        errors.append(
            "Thiếu các cột bắt buộc: " + ", ".join(missing_columns)
        )
        return errors, warnings

    columns_with_missing_values = [
        column
        for column in required_columns
        if dataframe[column].isna().any()
    ]
    if columns_with_missing_values:
        errors.append(
            "Các cột bắt buộc có dữ liệu trống: "
            + ", ".join(columns_with_missing_values)
        )

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        if column not in required_columns:
            continue

        numeric_values = pd.to_numeric(dataframe[column], errors="coerce")
        if numeric_values.isna().any():
            errors.append(f"Cột {column} phải chỉ chứa giá trị số.")
            continue

        if not numeric_values.between(minimum, maximum).all():
            errors.append(
                f"Cột {column} phải nằm trong khoảng {minimum}–{maximum}."
            )

    for column, allowed_values in CATEGORICAL_VALUES.items():
        if column not in required_columns:
            continue

        values = set(dataframe[column].astype(str))
        invalid_values = sorted(values - allowed_values)
        if invalid_values:
            preview = ", ".join(invalid_values[:5])
            errors.append(
                f"Cột {column} có giá trị không hợp lệ: {preview}."
            )

    if "G3" in dataframe.columns:
        warnings.append(
            "File có cột G3. Cột này chỉ được giữ để đối chiếu và không được đưa vào model."
        )

    return errors, warnings


def extract_api_error(response: requests.Response) -> Any:
    """Lấy nội dung lỗi FastAPI theo dạng dễ hiển thị."""

    try:
        payload = response.json()
    except ValueError:
        return response.text or "API không trả về nội dung lỗi."

    return payload.get("detail", payload)


def build_output_filename(original_filename: str) -> str:
    """Tạo tên file kết quả phía giao diện."""

    return f"{Path(original_filename).stem}_predictions.csv"


st.title("📁 Dự đoán hàng loạt")
st.caption(
    "Tải file CSV, gửi dữ liệu đến FastAPI và nhận lại file có điểm G3 dự đoán."
)

prediction_mode_label = st.radio(
    "Chế độ dự đoán",
    options=list(MODE_LABEL_TO_VALUE),
    horizontal=True,
)
prediction_mode = MODE_LABEL_TO_VALUE[prediction_mode_label]

if prediction_mode == "with_g1_g2":
    st.markdown(
        """
        <div class="recommend-note">
            <strong>✅ Chế độ khuyến nghị</strong><br>
            File phải có đủ G1 và G2. Model thường cho kết quả chính xác hơn.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="advanced-note">
            <strong>🔎 Chế độ dự báo sớm</strong><br>
            File không cần G1 và G2.
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("📘 Hướng dẫn chuẩn bị file CSV", expanded=False):
    st.markdown(
        """
        - Mỗi dòng tương ứng với một học sinh.
        - Tên cột phải giữ đúng như dữ liệu huấn luyện.
        - File dùng dấu phẩy và mã hóa UTF-8.
        - Không cần cột `G3`.
        - Giới hạn dung lượng gửi lên API là 10 MB.
        """
    )
    st.code(",".join(get_required_columns(prediction_mode)), language="text")

st.divider()

uploaded_file = st.file_uploader(
    "Chọn file CSV",
    type=["csv"],
    help="Mỗi dòng là một học sinh.",
)

if uploaded_file is None:
    st.info("Vui lòng tải lên một file CSV để bắt đầu.")
    st.stop()

try:
    uploaded_bytes = uploaded_file.getvalue()
    batch_data = pd.read_csv(BytesIO(uploaded_bytes))
except pd.errors.EmptyDataError:
    st.error("File CSV không chứa dữ liệu.")
    st.stop()
except pd.errors.ParserError:
    st.error("Không thể đọc cấu trúc CSV. Hãy kiểm tra dấu phân cách.")
    st.stop()
except UnicodeDecodeError:
    st.error("File phải sử dụng mã hóa UTF-8.")
    st.stop()
except Exception as error:
    st.error(f"Không thể đọc file CSV: {error}")
    st.stop()

if batch_data.empty:
    st.error("File CSV không có dòng học sinh nào.")
    st.stop()

summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Số học sinh", int(batch_data.shape[0]))
summary_col2.metric("Số cột", int(batch_data.shape[1]))
summary_col3.metric("Giá trị bị thiếu", int(batch_data.isna().sum().sum()))

st.subheader("👀 Xem trước dữ liệu")
st.dataframe(batch_data.head(10), use_container_width=True, hide_index=True)

validation_errors, validation_warnings = validate_uploaded_dataframe(
    batch_data,
    prediction_mode,
)

st.subheader("✅ Kiểm tra dữ liệu")
for warning in validation_warnings:
    st.warning(warning)
for error in validation_errors:
    st.error(error)

file_is_valid = not validation_errors
if file_is_valid:
    st.success("File hợp lệ và sẵn sàng gửi đến model.")
else:
    st.warning("Hãy sửa các lỗi trong file rồi tải lại.")

request_key = (
    f"{uploaded_file.name}:{len(uploaded_bytes)}:"
    f"{prediction_mode}:{hash(uploaded_bytes)}"
)

predict_button_clicked = st.button(
    "🤖 Dự đoán toàn bộ file",
    use_container_width=True,
    disabled=not file_is_valid,
)

if predict_button_clicked:
    try:
        with st.spinner("Đang gửi CSV đến FastAPI và chạy dự đoán..."):
            response = requests.post(
                API_URL,
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_bytes,
                        "text/csv",
                    )
                },
                data={
                    "prediction_mode": prediction_mode,
                    "include_input": "true",
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        if response.status_code == 200:
            result_dataframe = pd.read_csv(BytesIO(response.content))
            expected_columns = OUTPUT_COLUMNS
            missing_output_columns = expected_columns - set(result_dataframe.columns)
            if missing_output_columns:
                raise ValueError(
                    "API trả về CSV thiếu cột: "
                    + ", ".join(sorted(missing_output_columns))
                )

            st.session_state["batch_prediction_result"] = {
                "request_key": request_key,
                "csv_bytes": response.content,
                "filename": build_output_filename(uploaded_file.name),
            }
            st.success(
                f"Dự đoán thành công {len(result_dataframe)} học sinh."
            )
        else:
            st.error(f"API trả về lỗi {response.status_code}.")
            error_detail = extract_api_error(response)
            if isinstance(error_detail, dict):
                st.json(error_detail)
            elif isinstance(error_detail, list):
                for item in error_detail:
                    st.write(f"- {item}")
            else:
                st.code(str(error_detail))

    except requests.exceptions.ConnectionError:
        st.error(
            "Không kết nối được FastAPI. Hãy chạy API tại "
            f"{API_BASE_URL} trước."
        )
    except requests.exceptions.Timeout:
        st.error("API phản hồi quá lâu. Vui lòng thử lại.")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error("Không thể đọc file kết quả do API trả về.")
        st.code(str(error))
    except Exception as error:
        st.error("Đã xảy ra lỗi khi dự đoán hàng loạt.")
        st.code(str(error))

stored_result = st.session_state.get("batch_prediction_result")
if stored_result and stored_result.get("request_key") == request_key:
    result_dataframe = pd.read_csv(BytesIO(stored_result["csv_bytes"]))

    st.divider()
    st.subheader("🎯 Kết quả dự đoán")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Số dòng", len(result_dataframe))
    metric_col2.metric(
        "G3 trung bình",
        f"{result_dataframe['predicted_g3'].mean():.2f}",
    )
    metric_col3.metric(
        "G3 thấp nhất",
        f"{result_dataframe['predicted_g3'].min():.2f}",
    )
    metric_col4.metric(
        "G3 cao nhất",
        f"{result_dataframe['predicted_g3'].max():.2f}",
    )

    st.dataframe(
        result_dataframe.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Tải file kết quả CSV",
        data=stored_result["csv_bytes"],
        file_name=stored_result["filename"],
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("🛠 Thông tin kỹ thuật"):
        st.json(
            {
                "endpoint": API_URL,
                "prediction_mode": prediction_mode,
                "input_rows": int(batch_data.shape[0]),
                "output_rows": int(result_dataframe.shape[0]),
                "output_columns": result_dataframe.columns.tolist(),
            }
        )
