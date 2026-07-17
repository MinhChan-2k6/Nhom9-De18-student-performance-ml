from datetime import datetime

import requests
import streamlit as st

from streamlit_app.components.styles import load_css


# --------------------------------------------------
# Cấu hình trang
# --------------------------------------------------
st.set_page_config(
    page_title="Dự đoán kết quả",
    page_icon="🤖",
    layout="wide"
)

load_css()


# --------------------------------------------------
# Cấu hình API
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"


# --------------------------------------------------
# Các hàm hỗ trợ hiển thị
# --------------------------------------------------
def yes_no_label(value: str) -> str:
    """Chuyển yes/no thành Có/Không trên giao diện."""
    return {
        "yes": "Có",
        "no": "Không"
    }[value]


def education_label(value: int) -> str:
    """Hiển thị trình độ học vấn của cha hoặc mẹ."""
    return {
        0: "Không có",
        1: "Tiểu học",
        2: "Trung học cơ sở",
        3: "Trung học phổ thông",
        4: "Đại học hoặc cao hơn"
    }[value]


def job_label(value: str) -> str:
    """Hiển thị nghề nghiệp bằng tiếng Việt."""
    return {
        "teacher": "Giáo viên",
        "health": "Y tế",
        "services": "Dịch vụ",
        "at_home": "Nội trợ / làm việc tại nhà",
        "other": "Khác"
    }[value]


def level_label(value: int) -> str:
    """Hiển thị thang đánh giá từ 1 đến 5."""
    return {
        1: "1 - Rất thấp",
        2: "2 - Thấp",
        3: "3 - Trung bình",
        4: "4 - Tốt",
        5: "5 - Rất tốt"
    }[value]


# --------------------------------------------------
# Tiêu đề trang
# --------------------------------------------------
st.title("🤖 Dự đoán kết quả học tập")
st.caption(
    "Nhập thông tin học sinh để dự đoán điểm cuối kỳ G3."
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
    horizontal=True,
    help=(
        "Chế độ có G1 và G2 thường cho kết quả chính xác hơn. "
        "Chế độ dự báo sớm không sử dụng hai điểm này."
    )
)

if prediction_mode == "Có điểm G1 và G2":
    st.markdown(
        """
        <div class="recommend-note">
            <strong>✅ Chế độ khuyến nghị</strong><br>
            Sử dụng G1 và G2 thường giúp mô hình dự đoán điểm G3 chính xác hơn.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="advanced-note">
            <strong>🔎 Chế độ dự báo sớm</strong><br>
            Hệ thống dự đoán dựa trên các yếu tố cá nhân, gia đình và học tập
            mà không sử dụng G1, G2.
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# --------------------------------------------------
# Chia khu vực nhập liệu và kết quả
# --------------------------------------------------
input_column, result_column = st.columns(
    [3, 2],
    gap="large"
)


# --------------------------------------------------
# Khu vực nhập dữ liệu
# --------------------------------------------------
with input_column:
    st.markdown(
        """
        <div class="section-title">
            <h3>📝 Thông tin đầu vào</h3>
            <p>Điền đầy đủ các trường có dấu (*) trước khi dự đoán.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):
        basic_column, study_column = st.columns(
            [1, 1],
            gap="medium"
        )

        # ==========================================
        # THÔNG TIN CÁ NHÂN
        # ==========================================
        with basic_column:
            st.markdown(
                '<div class="form-group-heading">👤 Thông tin cá nhân</div>',
                unsafe_allow_html=True
            )

            full_name = st.text_input(
                "Họ và tên (*)",
                placeholder="Ví dụ: Nguyễn Văn An",
                max_chars=100
            )

            subject = st.selectbox(
                "Môn học (*)",
                options=["math", "portuguese"],
                index=None,
                placeholder="Chọn môn học",
                format_func=lambda value: {
                    "math": "Toán",
                    "portuguese": "Tiếng Bồ Đào Nha"
                }[value]
            )

            sex = st.selectbox(
                "Giới tính (*)",
                options=["f", "m"],
                index=None,
                placeholder="Chọn giới tính",
                format_func=lambda value: {
                    "f": "Nữ",
                    "m": "Nam"
                }[value]
            )

            age = st.selectbox(
                "Tuổi (*)",
                options=list(range(15, 23)),
                index=None,
                placeholder="Chọn tuổi",
                help="Độ tuổi trong dữ liệu hiện tại từ 15 đến 22."
            )

            studytime = st.selectbox(
                "Thời gian tự học mỗi tuần (*)",
                options=[1, 2, 3, 4],
                index=None,
                placeholder="Chọn thời gian tự học",
                format_func=lambda value: {
                    1: "Dưới 2 giờ",
                    2: "Từ 2 đến 5 giờ",
                    3: "Từ 5 đến 10 giờ",
                    4: "Trên 10 giờ"
                }[value]
            )

        # ==========================================
        # THÔNG TIN HỌC TẬP
        # ==========================================
        with study_column:
            st.markdown(
                '<div class="form-group-heading">📚 Thông tin học tập</div>',
                unsafe_allow_html=True
            )

            failures = st.selectbox(
                "Số lần trượt trước đây (*)",
                options=[0, 1, 2, 3],
                index=None,
                placeholder="Chọn số lần trượt"
            )

            absences_text = st.text_input(
                "Số buổi vắng (*)",
                placeholder="Nhập số từ 0 đến 75",
                help="Giới hạn dựa trên dữ liệu hiện tại."
            )

            internet = st.selectbox(
                "Có Internet tại nhà (*)",
                options=["yes", "no"],
                index=None,
                placeholder="Chọn câu trả lời",
                format_func=yes_no_label
            )

            family_support = st.selectbox(
                "Gia đình hỗ trợ học tập (*)",
                options=["yes", "no"],
                index=None,
                placeholder="Chọn câu trả lời",
                format_func=yes_no_label
            )

            g1 = None
            g2 = None

            if prediction_mode == "Có điểm G1 và G2":
                g1 = st.selectbox(
                    "Điểm G1 (*)",
                    options=list(range(0, 21)),
                    index=None,
                    placeholder="Chọn điểm G1",
                    help="Điểm G1 theo thang 0–20 của dữ liệu hiện tại."
                )

                g2 = st.selectbox(
                    "Điểm G2 (*)",
                    options=list(range(0, 21)),
                    index=None,
                    placeholder="Chọn điểm G2",
                    help="Điểm G2 theo thang 0–20 của dữ liệu hiện tại."
                )

        # ==========================================
        # THÔNG TIN NÂNG CAO
        # ==========================================
        st.markdown(
            """
            <div class="advanced-note">
                <strong>🧠 Thông tin nâng cao</strong><br>
                Các đặc trưng bổ sung giúp mô hình hiểu rõ hơn hoàn cảnh học tập
                và có thể cải thiện chất lượng dự đoán.
                Hệ thống không tự gán dữ liệu mặc định cho các trường này.
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            "Mở phần thông tin nâng cao",
            expanded=False
        ):
            advanced_col1, advanced_col2 = st.columns(
                [1, 1],
                gap="medium"
            )

            # --------------------------------------
            # Trường học và gia đình
            # --------------------------------------
            with advanced_col1:
                st.markdown(
                    '<div class="form-group-heading">🏠 Trường học và gia đình</div>',
                    unsafe_allow_html=True
                )

                school = st.selectbox(
                    "Trường (*)",
                    options=["gp", "ms"],
                    index=None,
                    placeholder="Chọn trường",
                    format_func=lambda value: {
                        "gp": "Gabriel Pereira (GP)",
                        "ms": "Mousinho da Silveira (MS)"
                    }[value]
                )

                address = st.selectbox(
                    "Khu vực sinh sống (*)",
                    options=["u", "r"],
                    index=None,
                    placeholder="Chọn khu vực",
                    format_func=lambda value: {
                        "u": "Thành thị",
                        "r": "Nông thôn"
                    }[value]
                )

                family_size = st.selectbox(
                    "Quy mô gia đình (*)",
                    options=["le3", "gt3"],
                    index=None,
                    placeholder="Chọn quy mô gia đình",
                    format_func=lambda value: {
                        "le3": "Từ 3 người trở xuống",
                        "gt3": "Trên 3 người"
                    }[value]
                )

                parent_status = st.selectbox(
                    "Tình trạng sống chung của cha mẹ (*)",
                    options=["t", "a"],
                    index=None,
                    placeholder="Chọn tình trạng",
                    format_func=lambda value: {
                        "t": "Sống cùng nhau",
                        "a": "Sống riêng"
                    }[value]
                )

                mother_education = st.selectbox(
                    "Trình độ học vấn của mẹ (*)",
                    options=[0, 1, 2, 3, 4],
                    index=None,
                    placeholder="Chọn trình độ",
                    format_func=education_label
                )

                father_education = st.selectbox(
                    "Trình độ học vấn của cha (*)",
                    options=[0, 1, 2, 3, 4],
                    index=None,
                    placeholder="Chọn trình độ",
                    format_func=education_label
                )

                mother_job = st.selectbox(
                    "Nghề nghiệp của mẹ (*)",
                    options=[
                        "teacher",
                        "health",
                        "services",
                        "at_home",
                        "other"
                    ],
                    index=None,
                    placeholder="Chọn nghề nghiệp",
                    format_func=job_label
                )

                father_job = st.selectbox(
                    "Nghề nghiệp của cha (*)",
                    options=[
                        "teacher",
                        "health",
                        "services",
                        "at_home",
                        "other"
                    ],
                    index=None,
                    placeholder="Chọn nghề nghiệp",
                    format_func=job_label
                )

                guardian = st.selectbox(
                    "Người giám hộ (*)",
                    options=["mother", "father", "other"],
                    index=None,
                    placeholder="Chọn người giám hộ",
                    format_func=lambda value: {
                        "mother": "Mẹ",
                        "father": "Cha",
                        "other": "Người khác"
                    }[value]
                )

                family_relation = st.selectbox(
                    "Mức độ quan hệ gia đình (*)",
                    options=[1, 2, 3, 4, 5],
                    index=None,
                    placeholder="Chọn mức độ",
                    format_func=level_label
                )

            # --------------------------------------
            # Học tập và sinh hoạt
            # --------------------------------------
            with advanced_col2:
                st.markdown(
                    '<div class="form-group-heading">🌱 Học tập và sinh hoạt</div>',
                    unsafe_allow_html=True
                )

                school_reason = st.selectbox(
                    "Lý do chọn trường (*)",
                    options=[
                        "course",
                        "home",
                        "reputation",
                        "other"
                    ],
                    index=None,
                    placeholder="Chọn lý do",
                    format_func=lambda value: {
                        "course": "Chương trình học",
                        "home": "Gần nhà",
                        "reputation": "Danh tiếng trường",
                        "other": "Lý do khác"
                    }[value]
                )

                travel_time = st.selectbox(
                    "Thời gian di chuyển đến trường (*)",
                    options=[1, 2, 3, 4],
                    index=None,
                    placeholder="Chọn thời gian",
                    format_func=lambda value: {
                        1: "Dưới 15 phút",
                        2: "Từ 15 đến 30 phút",
                        3: "Từ 30 phút đến 1 giờ",
                        4: "Trên 1 giờ"
                    }[value]
                )

                school_support = st.selectbox(
                    "Có hỗ trợ học tập từ trường (*)",
                    options=["yes", "no"],
                    index=None,
                    placeholder="Chọn câu trả lời",
                    format_func=yes_no_label
                )

                paid_classes = st.selectbox(
                    "Có học thêm trả phí (*)",
                    options=["yes", "no"],
                    index=None,
                    placeholder="Chọn câu trả lời",
                    format_func=yes_no_label
                )

                activities = st.selectbox(
                    "Có tham gia ngoại khóa (*)",
                    options=["yes", "no"],
                    index=None,
                    placeholder="Chọn câu trả lời",
                    format_func=yes_no_label
                )

                nursery = st.selectbox(
                    "Đã từng học mẫu giáo (*)",
                    options=["yes", "no"],
                    index=None,
                    placeholder="Chọn câu trả lời",
                    format_func=yes_no_label
                )

                higher = st.selectbox(
                    "Có mong muốn học lên cao (*)",
                    options=["yes", "no"],
                    index=None,
                    placeholder="Chọn câu trả lời",
                    format_func=yes_no_label
                )

                romantic = st.selectbox(
                    "Đang có mối quan hệ tình cảm (*)",
                    options=["yes", "no"],
                    index=None,
                    placeholder="Chọn câu trả lời",
                    format_func=yes_no_label
                )

                free_time = st.selectbox(
                    "Mức độ thời gian rảnh (*)",
                    options=[1, 2, 3, 4, 5],
                    index=None,
                    placeholder="Chọn mức độ",
                    format_func=level_label
                )

                going_out = st.selectbox(
                    "Mức độ đi chơi với bạn bè (*)",
                    options=[1, 2, 3, 4, 5],
                    index=None,
                    placeholder="Chọn mức độ",
                    format_func=level_label
                )

                weekday_alcohol = st.selectbox(
                    "Mức tiêu thụ đồ uống có cồn ngày thường (*)",
                    options=[1, 2, 3, 4, 5],
                    index=None,
                    placeholder="Chọn mức độ",
                    format_func=level_label
                )

                weekend_alcohol = st.selectbox(
                    "Mức tiêu thụ đồ uống có cồn cuối tuần (*)",
                    options=[1, 2, 3, 4, 5],
                    index=None,
                    placeholder="Chọn mức độ",
                    format_func=level_label
                )

                health = st.selectbox(
                    "Tình trạng sức khỏe (*)",
                    options=[1, 2, 3, 4, 5],
                    index=None,
                    placeholder="Chọn mức độ",
                    format_func=level_label
                )

        submitted = st.form_submit_button(
            "🤖 Dự đoán điểm G3",
            use_container_width=True
        )


# --------------------------------------------------
# Khu vực kết quả trước khi dự đoán
# --------------------------------------------------
with result_column:
    st.markdown(
        """
        <div class="section-title">
            <h3>🎯 Kết quả dự đoán</h3>
            <p>Điểm G3 là đầu ra chính của mô hình hồi quy.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not submitted:
        st.info(
            "Kết quả sẽ hiển thị tại đây sau khi nhập đầy đủ "
            "thông tin và nhấn Dự đoán."
        )

        result_card_html = """
<div class="prediction-result-card">
    <div class="prediction-result-icon">🎯</div>
    <p class="prediction-result-label">Điểm G3 dự đoán</p>
    <div class="prediction-result-score">--</div>
    <span class="prediction-result-status">Chưa dự đoán</span>
</div>
"""

        st.markdown(
            result_card_html,
            unsafe_allow_html=True
        )

        st.write(
            "**Chế độ:**",
            prediction_mode
        )


# --------------------------------------------------
# Validation và chuẩn bị dữ liệu
# --------------------------------------------------
if submitted:
    validation_errors = []

    cleaned_name = full_name.strip()
    absences = None

    # Kiểm tra họ tên
    if not cleaned_name:
        validation_errors.append(
            "Vui lòng nhập họ và tên học sinh."
        )
    elif len(cleaned_name) < 2:
        validation_errors.append(
            "Họ và tên phải có ít nhất 2 ký tự."
        )

    # Kiểm tra số buổi vắng
    if not absences_text.strip():
        validation_errors.append(
            "Vui lòng nhập số buổi vắng."
        )
    else:
        try:
            absences = int(absences_text)

            if absences < 0 or absences > 75:
                validation_errors.append(
                    "Số buổi vắng phải nằm trong khoảng từ 0 đến 75."
                )

        except ValueError:
            validation_errors.append(
                "Số buổi vắng phải là số nguyên."
            )

    # Kiểm tra các trường cơ bản
    basic_required_fields = {
        "Môn học": subject,
        "Giới tính": sex,
        "Tuổi": age,
        "Thời gian tự học": studytime,
        "Số lần trượt": failures,
        "Internet tại nhà": internet,
        "Hỗ trợ gia đình": family_support
    }

    for field_name, field_value in basic_required_fields.items():
        if field_value is None:
            validation_errors.append(
                f"Vui lòng nhập hoặc chọn {field_name.lower()}."
            )

    # Kiểm tra G1 và G2
    if prediction_mode == "Có điểm G1 và G2":
        if g1 is None:
            validation_errors.append(
                "Vui lòng chọn điểm G1."
            )

        if g2 is None:
            validation_errors.append(
                "Vui lòng chọn điểm G2."
            )

    # Kiểm tra thông tin nâng cao
    advanced_required_fields = {
        "Trường": school,
        "Khu vực sinh sống": address,
        "Quy mô gia đình": family_size,
        "Tình trạng cha mẹ": parent_status,
        "Học vấn của mẹ": mother_education,
        "Học vấn của cha": father_education,
        "Nghề nghiệp của mẹ": mother_job,
        "Nghề nghiệp của cha": father_job,
        "Người giám hộ": guardian,
        "Quan hệ gia đình": family_relation,
        "Lý do chọn trường": school_reason,
        "Thời gian di chuyển": travel_time,
        "Hỗ trợ từ trường": school_support,
        "Học thêm trả phí": paid_classes,
        "Hoạt động ngoại khóa": activities,
        "Học mẫu giáo": nursery,
        "Mong muốn học lên cao": higher,
        "Quan hệ tình cảm": romantic,
        "Thời gian rảnh": free_time,
        "Mức độ đi chơi": going_out,
        "Đồ uống có cồn ngày thường": weekday_alcohol,
        "Đồ uống có cồn cuối tuần": weekend_alcohol,
        "Sức khỏe": health
    }

    for field_name, field_value in advanced_required_fields.items():
        if field_value is None:
            validation_errors.append(
                f"Vui lòng hoàn thành trường: {field_name}."
            )

    # Hiển thị lỗi
    if validation_errors:
        with result_column:
            st.error(
                f"Dữ liệu chưa hợp lệ. "
                f"Còn {len(validation_errors)} mục cần kiểm tra."
            )

            for error_message in validation_errors:
                st.write(f"- {error_message}")

            st.warning(
                "Hãy mở phần Thông tin nâng cao "
                "để kiểm tra các trường còn thiếu."
            )

    # Chuẩn bị payload khi dữ liệu hợp lệ
    else:
        student_information = {
            "full_name": cleaned_name
        }

        model_input = {
            "subject": subject,
            "school": school,
            "sex": sex,
            "age": int(age),
            "address": address,
            "famsize": family_size,
            "Pstatus": parent_status,
            "Medu": int(mother_education),
            "Fedu": int(father_education),
            "Mjob": mother_job,
            "Fjob": father_job,
            "reason": school_reason,
            "guardian": guardian,
            "traveltime": int(travel_time),
            "studytime": int(studytime),
            "failures": int(failures),
            "schoolsup": school_support,
            "famsup": family_support,
            "paid": paid_classes,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": int(family_relation),
            "freetime": int(free_time),
            "goout": int(going_out),
            "Dalc": int(weekday_alcohol),
            "Walc": int(weekend_alcohol),
            "health": int(health),
            "absences": int(absences)
        }

        if prediction_mode == "Có điểm G1 và G2":
            model_input["G1"] = int(g1)
            model_input["G2"] = int(g2)
            prediction_mode_value = "with_g1_g2"
        else:
            prediction_mode_value = "early_prediction"

        request_payload = {
            "student": student_information,
            "prediction_mode": prediction_mode_value,
            "features": model_input
        }

        with result_column:
            try:
                with st.spinner("Đang gửi dữ liệu đến mô hình..."):
                    response = requests.post(
                        API_URL,
                        json=request_payload,
                        timeout=30
                    )

                if response.status_code == 200:
                    prediction_data = response.json()

                    predicted_g3 = float(
                        prediction_data["predicted_g3"]
                    )

                    converted_result = prediction_data.get(
                        "converted_result",
                        "Đạt" if predicted_g3 >= 10 else "Trượt"
                    )

                    model_name = prediction_data.get(
                        "model",
                        "Không xác định"
                    )

                    scenario_name = prediction_data.get(
                        "scenario",
                        prediction_mode_value
                    )

                    status_class = (
                        "prediction-pass"
                        if converted_result == "Đạt"
                        else "prediction-fail"
                    )

                    result_card_html = f"""
<div class="prediction-result-card">
    <div class="prediction-result-icon">🎯</div>
    <p class="prediction-result-label">Điểm G3 dự đoán</p>
    <div class="prediction-result-score">{predicted_g3:.2f}</div>
    <span class="prediction-result-status {status_class}">
        Kết quả quy đổi: {converted_result}
    </span>
</div>
"""

                    st.success("Dự đoán thành công.")
                    st.markdown(
                        result_card_html,
                        unsafe_allow_html=True
                    )

                    st.write("**Học sinh:**", cleaned_name)
                    st.write("**Chế độ:**", prediction_mode)
                    st.write("**Model:**", model_name)
                    st.caption(
                        "Đạt/Trượt chỉ là kết quả quy đổi từ điểm G3. "
                        "Mô hình chính vẫn là mô hình hồi quy."
                    )

                    if "prediction_history" not in st.session_state:
                        st.session_state.prediction_history = []

                    st.session_state.prediction_history.append(
                        {
                            "Họ tên": cleaned_name,
                            "Môn học": (
                                "Toán"
                                if subject == "math"
                                else "Tiếng Bồ Đào Nha"
                            ),
                            "Chế độ dự đoán": prediction_mode,
                            "Điểm G3 dự đoán": round(predicted_g3, 2),
                            "Kết quả quy đổi": converted_result,
                            "Model": model_name,
                            "Scenario": scenario_name,
                            "Thời gian": datetime.now().strftime(
                                "%d/%m/%Y %H:%M:%S"
                            )
                        }
                    )

                elif response.status_code == 422:
                    st.error(
                        "API từ chối dữ liệu đầu vào. "
                        "Hãy kiểm tra lại các trường đã nhập."
                    )

                    with st.expander(
                        "Chi tiết lỗi xác thực từ API"
                    ):
                        st.json(response.json())

                else:
                    st.error(
                        f"API trả về lỗi {response.status_code}."
                    )

                    with st.expander(
                        "Chi tiết lỗi từ API"
                    ):
                        try:
                            st.json(response.json())
                        except ValueError:
                            st.code(response.text)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Không kết nối được FastAPI. "
                    "Hãy chạy API tại http://127.0.0.1:8000 trước."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "API phản hồi quá lâu. Vui lòng thử lại."
                )

            except (KeyError, TypeError, ValueError) as error:
                st.error(
                    "Phản hồi từ API không đúng định dạng mong đợi."
                )

                with st.expander(
                    "Chi tiết lỗi dành cho nhóm phát triển"
                ):
                    st.code(str(error))

        with st.expander(
            "🛠 Dữ liệu kỹ thuật dành cho nhóm phát triển"
        ):
            st.json(request_payload)
