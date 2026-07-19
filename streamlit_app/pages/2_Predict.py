from __future__ import annotations

import requests
import streamlit as st
from streamlit_app.components.styles import load_css
from streamlit_app.config import API_BASE_URL


# --------------------------------------------------
# Cấu hình trang
# --------------------------------------------------
st.set_page_config(
    page_title="Dự đoán điểm G3",
    page_icon="🤖",
    layout="wide",
)

load_css()


API_URL = f"{API_BASE_URL}/predict"


# --------------------------------------------------
# Các hàm hỗ trợ hiển thị
# --------------------------------------------------
def yes_no_label(value: str) -> str:
    """Chuyển yes/no thành Có/Không trên giao diện."""

    return {
        "yes": "Có",
        "no": "Không",
    }[value]


def education_label(value: int) -> str:
    """Hiển thị trình độ học vấn của cha hoặc mẹ."""

    return {
        0: "Không có",
        1: "Tiểu học",
        2: "Trung học cơ sở",
        3: "Trung học phổ thông",
        4: "Đại học hoặc cao hơn",
    }[value]


def job_label(value: str) -> str:
    """Hiển thị nghề nghiệp bằng tiếng Việt."""

    return {
        "teacher": "Giáo viên",
        "health": "Y tế",
        "services": "Dịch vụ",
        "at_home": "Nội trợ / làm việc tại nhà",
        "other": "Khác",
    }[value]


def level_label(value: int) -> str:
    """Hiển thị thang đánh giá từ 1 đến 5."""

    return {
        1: "1 - Rất thấp",
        2: "2 - Thấp",
        3: "3 - Trung bình",
        4: "4 - Tốt",
        5: "5 - Rất tốt",
    }[value]


def display_api_error(
    response: requests.Response,
) -> None:
    """Hiển thị nội dung lỗi trả về từ FastAPI."""

    st.error(
        f"API trả về lỗi {response.status_code}."
    )

    with st.expander(
        "Chi tiết lỗi từ API"
    ):
        try:
            st.json(
                response.json()
            )

        except ValueError:
            st.code(
                response.text
            )


# --------------------------------------------------
# Tiêu đề trang
# --------------------------------------------------
st.title(
    "🤖 Dự đoán điểm G3"
)

st.caption(
    "Hệ thống sử dụng mô hình hồi quy để dự đoán điểm G3. "
    "Chỉ cần nhập thông tin cơ bản; thông tin chi tiết là tùy chọn."
)


# --------------------------------------------------
# Chọn chế độ dự đoán
# --------------------------------------------------
prediction_mode = st.radio(
    "Chế độ dự đoán",
    options=[
        "Có điểm G1 và G2",
        "Dự báo sớm, không có G1 và G2",
    ],
    horizontal=True,
    help=(
        "Chế độ có G1 và G2 sử dụng thêm điểm học kỳ trước. "
        "Chế độ dự báo sớm không sử dụng hai điểm này."
    ),
)


if prediction_mode == "Có điểm G1 và G2":

    prediction_mode_value = (
        "with_g1_g2"
    )

    st.markdown(
        """
        <div class="recommend-note">
            <strong>
                ✅ Chế độ có G1 và G2
            </strong>
            <br>
            G1 và G2 là hai thông tin bắt buộc
            trong chế độ này.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    prediction_mode_value = (
        "early_prediction"
    )

    st.markdown(
        """
        <div class="advanced-note">
            <strong>
                🔎 Chế độ dự báo sớm
            </strong>
            <br>
            Hệ thống không sử dụng G1 và G2,
            cho phép ước lượng sớm điểm G3.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# --------------------------------------------------
# Chia khu vực nhập liệu và kết quả
# --------------------------------------------------
input_column, result_column = (
    st.columns(
        [3, 2],
        gap="large",
    )
)


# --------------------------------------------------
# Khu vực nhập dữ liệu
# --------------------------------------------------
with input_column:

    st.markdown(
        """
        <div class="section-title">
            <h3>
                📝 Thông tin đầu vào
            </h3>
            <p>
                Chỉ các trường có dấu (*) là bắt buộc.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form(
        "prediction_form"
    ):

        basic_column, study_column = (
            st.columns(
                [1, 1],
                gap="medium",
            )
        )

        # ==========================================
        # THÔNG TIN CƠ BẢN
        # ==========================================
        with basic_column:

            st.markdown(
                (
                    '<div class="form-group-heading">'
                    '👤 Thông tin cơ bản'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

            full_name = st.text_input(
                "Họ và tên (*)",
                placeholder=(
                    "Ví dụ: Nguyễn Văn An"
                ),
                max_chars=100,
            )

            subject = st.selectbox(
                "Môn học (*)",
                options=[
                    "math",
                    "portuguese",
                ],
                index=None,
                placeholder=(
                    "Chọn môn học"
                ),
                format_func=lambda value: {
                    "math": "Toán",
                    "portuguese":
                        "Tiếng Bồ Đào Nha",
                }[value],
            )

            sex = st.selectbox(
                "Giới tính (*)",
                options=[
                    "f",
                    "m",
                ],
                index=None,
                placeholder=(
                    "Chọn giới tính"
                ),
                format_func=lambda value: {
                    "f": "Nữ",
                    "m": "Nam",
                }[value],
            )

            age = st.selectbox(
                "Tuổi (*)",
                options=list(
                    range(
                        15,
                        23,
                    )
                ),
                index=None,
                placeholder=(
                    "Chọn tuổi"
                ),
                help=(
                    "Độ tuổi hợp lệ "
                    "từ 15 đến 22."
                ),
            )


        # ==========================================
        # THÔNG TIN HỌC TẬP BẮT BUỘC
        # ==========================================
        with study_column:

            st.markdown(
                (
                    '<div class="form-group-heading">'
                    '📚 Thông tin học tập'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

            studytime = st.selectbox(
                (
                    "Thời gian tự học "
                    "mỗi tuần (*)"
                ),
                options=[
                    1,
                    2,
                    3,
                    4,
                ],
                index=None,
                placeholder=(
                    "Chọn thời gian tự học"
                ),
                format_func=lambda value: {
                    1: "Dưới 2 giờ",
                    2: "Từ 2 đến 5 giờ",
                    3: "Từ 5 đến 10 giờ",
                    4: "Trên 10 giờ",
                }[value],
            )

            failures = st.selectbox(
                (
                    "Số lần học lại "
                    "trước đây (*)"
                ),
                options=[
                    0,
                    1,
                    2,
                    3,
                ],
                index=None,
                placeholder=(
                    "Chọn số lần học lại"
                ),
            )

            absences_text = (
                st.text_input(
                    "Số buổi vắng (*)",
                    placeholder=(
                        "Nhập số từ 0 đến 75"
                    ),
                    help=(
                        "Chỉ nhập số nguyên "
                        "từ 0 đến 75."
                    ),
                )
            )

            g1 = None
            g2 = None

            if (
                prediction_mode_value
                == "with_g1_g2"
            ):

                g1 = st.selectbox(
                    "Điểm G1 (*)",
                    options=list(
                        range(
                            0,
                            21,
                        )
                    ),
                    index=None,
                    placeholder=(
                        "Chọn điểm G1"
                    ),
                )

                g2 = st.selectbox(
                    "Điểm G2 (*)",
                    options=list(
                        range(
                            0,
                            21,
                        )
                    ),
                    index=None,
                    placeholder=(
                        "Chọn điểm G2"
                    ),
                )


        # ==========================================
        # THÔNG TIN CHI TIẾT TÙY CHỌN
        # ==========================================
        st.markdown(
            """
            <div class="advanced-note">
                <strong>
                    🧠 Thông tin chi tiết
                    — không bắt buộc
                </strong>
                <br>
                Khi bỏ trống, API sử dụng các giá trị
                đại diện đã cấu hình.
                Khi nhập, dữ liệu thật của học sinh
                sẽ được ưu tiên.
            </div>
            """,
            unsafe_allow_html=True,
        )


        with st.expander(
            "Mở phần thông tin chi tiết",
            expanded=False,
        ):

            advanced_col1, advanced_col2 = (
                st.columns(
                    [1, 1],
                    gap="medium",
                )
            )


            with advanced_col1:

                st.markdown(
                    (
                        '<div class="form-group-heading">'
                        '🏠 Trường học và gia đình'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

                school = st.selectbox(
                    "Trường",
                    options=[
                        "gp",
                        "ms",
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=lambda value: {
                        "gp":
                            "Gabriel Pereira (GP)",
                        "ms":
                            "Mousinho da Silveira (MS)",
                    }[value],
                )

                address = st.selectbox(
                    "Khu vực sinh sống",
                    options=[
                        "u",
                        "r",
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=lambda value: {
                        "u": "Thành thị",
                        "r": "Nông thôn",
                    }[value],
                )

                family_size = st.selectbox(
                    "Quy mô gia đình",
                    options=[
                        "le3",
                        "gt3",
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=lambda value: {
                        "le3":
                            "Từ 3 người trở xuống",
                        "gt3":
                            "Trên 3 người",
                    }[value],
                )

                parent_status = (
                    st.selectbox(
                        (
                            "Tình trạng sống chung "
                            "của cha mẹ"
                        ),
                        options=[
                            "t",
                            "a",
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=lambda value: {
                            "t":
                                "Sống cùng nhau",
                            "a":
                                "Sống riêng",
                        }[value],
                    )
                )

                mother_education = (
                    st.selectbox(
                        (
                            "Trình độ học vấn "
                            "của mẹ"
                        ),
                        options=[
                            0,
                            1,
                            2,
                            3,
                            4,
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            education_label
                        ),
                    )
                )

                father_education = (
                    st.selectbox(
                        (
                            "Trình độ học vấn "
                            "của cha"
                        ),
                        options=[
                            0,
                            1,
                            2,
                            3,
                            4,
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            education_label
                        ),
                    )
                )

                mother_job = (
                    st.selectbox(
                        (
                            "Nghề nghiệp "
                            "của mẹ"
                        ),
                        options=[
                            "teacher",
                            "health",
                            "services",
                            "at_home",
                            "other",
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            job_label
                        ),
                    )
                )

                father_job = (
                    st.selectbox(
                        (
                            "Nghề nghiệp "
                            "của cha"
                        ),
                        options=[
                            "teacher",
                            "health",
                            "services",
                            "at_home",
                            "other",
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            job_label
                        ),
                    )
                )

                guardian = st.selectbox(
                    "Người giám hộ",
                    options=[
                        "mother",
                        "father",
                        "other",
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=lambda value: {
                        "mother": "Mẹ",
                        "father": "Cha",
                        "other": "Người khác",
                    }[value],
                )

                family_support = (
                    st.selectbox(
                        (
                            "Gia đình hỗ trợ "
                            "học tập"
                        ),
                        options=[
                            "yes",
                            "no",
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            yes_no_label
                        ),
                    )
                )

                family_relation = (
                    st.selectbox(
                        (
                            "Mức độ quan hệ "
                            "gia đình"
                        ),
                        options=[
                            1,
                            2,
                            3,
                            4,
                            5,
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            level_label
                        ),
                    )
                )


            with advanced_col2:

                st.markdown(
                    (
                        '<div class="form-group-heading">'
                        '🌱 Học tập và sinh hoạt'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

                school_reason = (
                    st.selectbox(
                        "Lý do chọn trường",
                        options=[
                            "course",
                            "home",
                            "reputation",
                            "other",
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=lambda value: {
                            "course":
                                "Chương trình học",
                            "home":
                                "Gần nhà",
                            "reputation":
                                "Danh tiếng trường",
                            "other":
                                "Lý do khác",
                        }[value],
                    )
                )

                travel_time = (
                    st.selectbox(
                        (
                            "Thời gian di chuyển "
                            "đến trường"
                        ),
                        options=[
                            1,
                            2,
                            3,
                            4,
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=lambda value: {
                            1:
                                "Dưới 15 phút",
                            2:
                                "Từ 15 đến 30 phút",
                            3:
                                "Từ 30 phút đến 1 giờ",
                            4:
                                "Trên 1 giờ",
                        }[value],
                    )
                )

                school_support = (
                    st.selectbox(
                        (
                            "Có hỗ trợ học tập "
                            "từ trường"
                        ),
                        options=[
                            "yes",
                            "no",
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            yes_no_label
                        ),
                    )
                )

                paid_classes = (
                    st.selectbox(
                        (
                            "Có học thêm "
                            "trả phí"
                        ),
                        options=[
                            "yes",
                            "no",
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            yes_no_label
                        ),
                    )
                )

                activities = (
                    st.selectbox(
                        (
                            "Có tham gia "
                            "ngoại khóa"
                        ),
                        options=[
                            "yes",
                            "no",
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            yes_no_label
                        ),
                    )
                )

                nursery = st.selectbox(
                    (
                        "Đã từng học "
                        "mẫu giáo"
                    ),
                    options=[
                        "yes",
                        "no",
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=(
                        yes_no_label
                    ),
                )

                higher = st.selectbox(
                    (
                        "Có mong muốn "
                        "học lên cao"
                    ),
                    options=[
                        "yes",
                        "no",
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=(
                        yes_no_label
                    ),
                )

                internet = st.selectbox(
                    "Có Internet tại nhà",
                    options=[
                        "yes",
                        "no",
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=(
                        yes_no_label
                    ),
                )

                romantic = st.selectbox(
                    (
                        "Đang có mối quan hệ "
                        "tình cảm"
                    ),
                    options=[
                        "yes",
                        "no",
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=(
                        yes_no_label
                    ),
                )

                free_time = st.selectbox(
                    "Mức độ thời gian rảnh",
                    options=[
                        1,
                        2,
                        3,
                        4,
                        5,
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=(
                        level_label
                    ),
                )

                going_out = st.selectbox(
                    (
                        "Mức độ đi chơi "
                        "với bạn bè"
                    ),
                    options=[
                        1,
                        2,
                        3,
                        4,
                        5,
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=(
                        level_label
                    ),
                )

                weekday_alcohol = (
                    st.selectbox(
                        (
                            "Mức tiêu thụ đồ uống "
                            "có cồn ngày thường"
                        ),
                        options=[
                            1,
                            2,
                            3,
                            4,
                            5,
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            level_label
                        ),
                    )
                )

                weekend_alcohol = (
                    st.selectbox(
                        (
                            "Mức tiêu thụ đồ uống "
                            "có cồn cuối tuần"
                        ),
                        options=[
                            1,
                            2,
                            3,
                            4,
                            5,
                        ],
                        index=None,
                        placeholder=(
                            "Không bắt buộc"
                        ),
                        format_func=(
                            level_label
                        ),
                    )
                )

                health = st.selectbox(
                    (
                        "Tình trạng "
                        "sức khỏe"
                    ),
                    options=[
                        1,
                        2,
                        3,
                        4,
                        5,
                    ],
                    index=None,
                    placeholder=(
                        "Không bắt buộc"
                    ),
                    format_func=(
                        level_label
                    ),
                )


        submitted = (
            st.form_submit_button(
                "🤖 Dự đoán điểm G3",
                use_container_width=True,
            )
        )


# --------------------------------------------------
# Kết quả mặc định
# --------------------------------------------------
with result_column:

    st.markdown(
        """
        <div class="section-title">
            <h3>
                🎯 Kết quả dự đoán
            </h3>
            <p>
                Điểm G3 được dự đoán bởi mô hình hồi quy,
                kèm sai số tham khảo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not submitted:

        st.info(
            "Nhập các thông tin cơ bản "
            "có dấu (*) rồi nhấn Dự đoán."
        )

        st.markdown(
            """
            <div class="prediction-result-card">
                <div class="prediction-result-icon">
                    🎯
                </div>
                <p class="prediction-result-label">
                    Điểm G3 dự đoán
                </p>
                <div class="prediction-result-score">
                    --
                </div>
                <span class="prediction-result-status">
                    Chưa dự đoán
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "**Chế độ:**",
            prediction_mode,
        )


# --------------------------------------------------
# Validation, gửi API và hiển thị kết quả
# --------------------------------------------------
if submitted:

    validation_errors: list[str] = []

    cleaned_name = (
        full_name.strip()
    )

    absences: int | None = None


    if not cleaned_name:

        validation_errors.append(
            "Vui lòng nhập họ và tên học sinh."
        )

    elif len(
        cleaned_name
    ) < 2:

        validation_errors.append(
            "Họ và tên phải có ít nhất 2 ký tự."
        )


    required_fields = {
        "môn học": subject,
        "giới tính": sex,
        "tuổi": age,
        "thời gian tự học":
            studytime,
        "số lần học lại":
            failures,
    }


    for (
        field_name,
        field_value,
    ) in required_fields.items():

        if field_value is None:

            validation_errors.append(
                (
                    "Vui lòng nhập hoặc chọn "
                    f"{field_name}."
                )
            )


    if not absences_text.strip():

        validation_errors.append(
            "Vui lòng nhập số buổi vắng."
        )

    else:

        try:

            absences = int(
                absences_text
            )

            if (
                absences < 0
                or absences > 75
            ):

                validation_errors.append(
                    (
                        "Số buổi vắng phải nằm "
                        "trong khoảng từ 0 đến 75."
                    )
                )

        except ValueError:

            validation_errors.append(
                "Số buổi vắng phải là số nguyên."
            )


    if (
        prediction_mode_value
        == "with_g1_g2"
    ):

        if g1 is None:

            validation_errors.append(
                "Vui lòng chọn điểm G1."
            )

        if g2 is None:

            validation_errors.append(
                "Vui lòng chọn điểm G2."
            )


    with result_column:

        if validation_errors:

            st.error(
                (
                    "Dữ liệu chưa hợp lệ. "
                    f"Còn {len(validation_errors)} "
                    "mục cần kiểm tra."
                )
            )

            for error in (
                validation_errors
            ):

                st.write(
                    f"- {error}"
                )


        else:

            model_input = {
                "subject": subject,
                "sex": sex,
                "age": int(age),
                "studytime":
                    int(studytime),
                "failures":
                    int(failures),
                "absences":
                    int(absences),
            }


            optional_features = {
                "school":
                    school,
                "address":
                    address,
                "famsize":
                    family_size,
                "Pstatus":
                    parent_status,
                "Medu":
                    mother_education,
                "Fedu":
                    father_education,
                "Mjob":
                    mother_job,
                "Fjob":
                    father_job,
                "reason":
                    school_reason,
                "guardian":
                    guardian,
                "traveltime":
                    travel_time,
                "schoolsup":
                    school_support,
                "famsup":
                    family_support,
                "paid":
                    paid_classes,
                "activities":
                    activities,
                "nursery":
                    nursery,
                "higher":
                    higher,
                "internet":
                    internet,
                "romantic":
                    romantic,
                "famrel":
                    family_relation,
                "freetime":
                    free_time,
                "goout":
                    going_out,
                "Dalc":
                    weekday_alcohol,
                "Walc":
                    weekend_alcohol,
                "health":
                    health,
            }


            model_input.update(
                {
                    key: value
                    for (
                        key,
                        value,
                    )
                    in optional_features.items()
                    if value is not None
                }
            )


            if (
                prediction_mode_value
                == "with_g1_g2"
            ):

                model_input["G1"] = (
                    int(g1)
                )

                model_input["G2"] = (
                    int(g2)
                )


            request_payload = {
                "student": {
                    "full_name":
                        cleaned_name,
                },
                "prediction_mode":
                    prediction_mode_value,
                "features":
                    model_input,
            }


            try:

                with st.spinner(
                    (
                        "Đang gửi dữ liệu đến "
                        "mô hình hồi quy..."
                    )
                ):

                    response = (
                        requests.post(
                            API_URL,
                            json=(
                                request_payload
                            ),
                            timeout=30,
                        )
                    )


                if (
                    response.status_code
                    == 200
                ):

                    prediction_data = (
                        response.json()
                    )


                    predicted_g3 = float(
                        prediction_data[
                            "predicted_g3"
                        ]
                    )


                    mae = float(
                        prediction_data[
                            "mae"
                        ]
                    )


                    rmse = float(
                        prediction_data[
                            "rmse"
                        ]
                    )


                    prediction_lower = float(
                        prediction_data[
                            "prediction_lower"
                        ]
                    )


                    prediction_upper = float(
                        prediction_data[
                            "prediction_upper"
                        ]
                    )


                    model_name = (
                        prediction_data.get(
                            "model",
                            "Không xác định",
                        )
                    )


                    model_version = (
                        prediction_data.get(
                            "model_version",
                            "Không xác định",
                        )
                    )


                    scenario_name = (
                        prediction_data.get(
                            "scenario",
                            prediction_mode_value,
                        )
                    )


                    history_saved = bool(
                        prediction_data.get(
                            "history_saved",
                            False,
                        )
                    )


                    history_id = (
                        prediction_data.get(
                            "history_id"
                        )
                    )


                    st.success(
                        (
                            "Dự đoán điểm G3 "
                            "thành công."
                        )
                    )


                    # ==================================
                    # KẾT QUẢ DỰ ĐOÁN + SAI SỐ
                    # ==================================
                    st.markdown(
                        f"""
                        <div class="prediction-result-card">
                            <div class="prediction-result-icon">
                                🎯
                            </div>

                            <p class="prediction-result-label">
                                Điểm G3 dự đoán
                            </p>

                            <div class="prediction-result-score">
                                {predicted_g3:.2f} ± {rmse:.2f}
                            </div>

                            <span class="prediction-result-status">
                                Thang điểm 0 – 20
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                    # ==================================
                    # KHOẢNG SAI SỐ THAM KHẢO
                    # ==================================
                    st.markdown(
                        "#### 📏 Sai số dự đoán tham khảo"
                    )

                    st.info(
                        (
                            f"**Điểm dự đoán:** "
                            f"{predicted_g3:.2f} / 20\n\n"
                            f"**Sai số tham khảo:** "
                            f"±{rmse:.2f} điểm\n\n"
                            f"**Khoảng dự đoán:** "
                            f"{prediction_lower:.2f} – "
                            f"{prediction_upper:.2f} / 20"
                        )
                    )


                    # ==================================
                    # CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH
                    # ==================================
                    error_col1, error_col2 = (
                        st.columns(2)
                    )


                    with error_col1:

                        st.metric(
                            "MAE",
                            f"{mae:.2f} điểm",
                            help=(
                                "Sai số tuyệt đối trung bình "
                                "của mô hình trên tập đánh giá."
                            ),
                        )


                    with error_col2:

                        st.metric(
                            "RMSE",
                            f"{rmse:.2f} điểm",
                            help=(
                                "RMSE được dùng để tạo khoảng sai số "
                                "tham khảo quanh điểm G3 dự đoán."
                            ),
                        )


                    st.caption(
                        (
                            "⚠️ Khoảng tham khảo được tính bằng "
                            "G3 dự đoán ± RMSE và giới hạn trong "
                            "thang điểm 0–20. Đây là khoảng sai số "
                            "tham khảo, không phải khoảng tin cậy 95%."
                        )
                    )


                    st.divider()


                    # ==================================
                    # THÔNG TIN CHI TIẾT
                    # ==================================
                    st.write(
                        "**Học sinh:**",
                        cleaned_name,
                    )


                    st.write(
                        "**Điểm G3 dự đoán:**",
                        (
                            f"{predicted_g3:.2f} "
                            "/ 20"
                        ),
                    )


                    st.write(
                        "**Khoảng tham khảo:**",
                        (
                            f"{prediction_lower:.2f} "
                            f"– {prediction_upper:.2f}"
                        ),
                    )


                    st.write(
                        "**Model:**",
                        model_name,
                    )


                    st.write(
                        "**Phiên bản model:**",
                        model_version,
                    )

                    st.write(
                        "**Scenario:**",
                        scenario_name,
                    )


                    if history_saved:
                        history_message = (
                            "Giá trị dự đoán G3 đã được "
                            "lưu vào SQL Server."
                        )
                        if history_id is not None:
                            history_message += (
                                f" Mã lịch sử: {history_id}."
                            )
                        st.caption(history_message)
                    else:
                        st.warning(
                            "Dự đoán đã thành công nhưng chưa thể "
                            "lưu lịch sử vào SQL Server. Bạn vẫn "
                            "có thể sử dụng kết quả phía trên."
                        )


                else:

                    display_api_error(
                        response
                    )


            except (
                requests.exceptions.ConnectionError
            ):

                st.error(
                    (
                        "Không kết nối được FastAPI. "
                        "Hãy chạy API tại "
                        "http://127.0.0.1:8000 trước."
                    )
                )


            except (
                requests.exceptions.Timeout
            ):

                st.error(
                    (
                        "API phản hồi quá lâu. "
                        "Vui lòng thử lại."
                    )
                )


            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:

                st.error(
                    (
                        "Phản hồi từ API không đúng "
                        "định dạng mong đợi."
                    )
                )

                with st.expander(
                    (
                        "Chi tiết lỗi dành cho "
                        "nhóm phát triển"
                    )
                ):

                    st.code(
                        str(error)
                    )


            with st.expander(
                (
                    "🛠 Dữ liệu kỹ thuật "
                    "dành cho nhóm phát triển"
                )
            ):

                st.json(
                    request_payload
                )