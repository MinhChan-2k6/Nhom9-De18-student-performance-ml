from __future__ import annotations

import html
from textwrap import dedent
from typing import Any

import pandas as pd
import requests
import streamlit as st

from streamlit_app.components.styles import load_css
from streamlit_app.config import API_BASE_URL



# ==================================================
# CẤU HÌNH TRANG
# ==================================================
st.set_page_config(
    page_title="Thông tin mô hình",
    page_icon="🧠",
    layout="wide",
)

load_css()


# ==================================================
# HÀM RENDER HTML
# Tránh Streamlit hiểu HTML thụt dòng thành code block
# ==================================================
def render_html(content: str) -> None:
    """
    Render HTML trong Streamlit mà không bị Markdown
    hiểu nhầm các dòng thụt lề thành code block.
    """

    cleaned_html = "".join(
        line.strip()
        for line in dedent(content).splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True,
    )


# ==================================================
# CSS RIÊNG CHO MODEL INFO
# ==================================================
render_html(
    """
    <style>
    .model-hero {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        background:
            radial-gradient(
                circle at top right,
                rgba(255,255,255,.22),
                transparent 34%
            ),
            linear-gradient(
                135deg,
                #155EEF 0%,
                #5B4BDB 58%,
                #7A5AF8 100%
            );
        color: white;
        box-shadow: 0 14px 34px rgba(21, 94, 239, .18);
        margin-bottom: 1.4rem;
    }

    .model-hero h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.05;
        color: white;
    }

    .model-hero p {
        margin: .7rem 0 0;
        color: rgba(255,255,255,.88);
        font-size: 1.02rem;
    }

    .scenario-banner {
        padding: 1.1rem 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(21, 94, 239, .14);
        background: linear-gradient(
            135deg,
            rgba(21,94,239,.08),
            rgba(122,90,248,.05)
        );
        margin: .35rem 0 1rem;
    }

    .scenario-banner .eyebrow {
        color: #155EEF;
        font-weight: 800;
        font-size: .78rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .25rem;
    }

    .scenario-banner h3 {
        margin: 0;
        color: #172B4D;
        font-size: 1.35rem;
    }

    .scenario-banner p {
        margin: .45rem 0 0;
        color: #667085;
    }

    .info-card {
        height: 100%;
        padding: 1rem 1.1rem;
        border-radius: 17px;
        background: white;
        border: 1px solid #E4E7EC;
        box-shadow: 0 7px 20px rgba(16, 24, 40, .06);
    }

    .info-card .label {
        color: #667085;
        font-size: .82rem;
        font-weight: 700;
        margin-bottom: .38rem;
    }

    .info-card .value {
        color: #101828;
        font-size: 1.28rem;
        line-height: 1.25;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .info-card .note {
        color: #98A2B3;
        font-size: .78rem;
        margin-top: .3rem;
    }

    .metric-box {
        padding: 1.05rem 1.1rem;
        border-radius: 17px;
        background: #F8FAFC;
        border: 1px solid #EAECF0;
        text-align: center;
    }

    .metric-box .metric-name {
        color: #667085;
        font-size: .82rem;
        font-weight: 700;
    }

    .metric-box .metric-value {
        color: #155EEF;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: .2rem;
    }

    .metric-box .metric-help {
        color: #98A2B3;
        font-size: .72rem;
        margin-top: .15rem;
    }

    .feature-chip {
        display: inline-block;
        padding: .35rem .62rem;
        margin: .2rem .18rem;
        border-radius: 999px;
        background: #EEF4FF;
        color: #1849A9;
        border: 1px solid #D1E0FF;
        font-size: .78rem;
        font-weight: 650;
    }

    .section-heading {
        margin: 1.5rem 0 .75rem;
        padding-left: .85rem;
        border-left: 5px solid #155EEF;
    }

    .section-heading h3 {
        margin: 0;
        color: #172B4D;
    }

    .section-heading p {
        margin: .3rem 0 0;
        color: #667085;
    }

    .status-ready {
        display: inline-flex;
        align-items: center;
        gap: .38rem;
        padding: .4rem .72rem;
        border-radius: 999px;
        background: #ECFDF3;
        color: #027A48;
        border: 1px solid #ABEFC6;
        font-weight: 750;
        font-size: .82rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #EAECF0;
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
    """
)


# ==================================================
# GỌI API MODEL INFO
# ==================================================
@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def load_model_info() -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE_URL}/model-info",
        timeout=15,
    )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, list):
        raise ValueError(
            "API /model-info phải trả về một danh sách."
        )

    return payload


# ==================================================
# FORMAT TÊN MODEL
# ==================================================
def format_model_name(
    value: Any,
) -> str:
    if not value:
        return "Không xác định"

    labels = {
        "random_forest":
            "Random Forest",
        "random_forest_grid_search":
            "Random Forest — Grid Search",
        "gradient_boosting":
            "Gradient Boosting",
        "linear_regression":
            "Linear Regression",
    }

    raw = str(value)

    return labels.get(
        raw,
        raw.replace(
            "_",
            " ",
        ).title(),
    )


# ==================================================
# FORMAT SCENARIO
# ==================================================
def format_scenario(
    value: Any,
) -> str:
    labels = {
        "A_with_G1_G2":
            "Có điểm G1 và G2",
        "B_without_G1_G2":
            "Dự báo sớm — không dùng G1, G2",
    }

    raw = str(
        value or ""
    )

    return labels.get(
        raw,
        raw or "Không xác định",
    )


def format_scenario_short(
    value: Any,
) -> str:
    labels = {
        "A_with_G1_G2":
            "Kịch bản A",
        "B_without_G1_G2":
            "Kịch bản B",
    }

    raw = str(
        value or ""
    )

    return labels.get(
        raw,
        raw or "Không xác định",
    )


# ==================================================
# FORMAT FEATURE SET
# ==================================================
def format_feature_set(
    scenario: Any,
) -> str:
    labels = {
        "A_with_G1_G2":
            "Có G1, G2",
        "B_without_G1_G2":
            "Không có G1, G2",
    }

    raw = str(
        scenario or ""
    )

    return labels.get(
        raw,
        "Không xác định",
    )


# ==================================================
# FORMAT MODE
# ==================================================
def format_mode(
    value: Any,
) -> str:
    labels = {
        "with_g1_g2":
            "Có G1 và G2",
        "early_prediction":
            "Dự báo sớm",
    }

    raw = str(
        value or ""
    )

    return labels.get(
        raw,
        raw or "Không xác định",
    )


# ==================================================
# LẤY METRIC
# ==================================================
def get_best_metric(
    metrics: dict[str, Any],
    key: str,
    *,
    model_name: Any = None,
    scenario: Any = None,
) -> float | None:

    for container_name in (
        "test_metrics",
        "best_metrics",
        "metrics",
    ):
        container = metrics.get(
            container_name
        )

        if (
            isinstance(
                container,
                dict,
            )
            and container.get(key)
            is not None
        ):
            try:
                return float(
                    container[key]
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

    if metrics.get(key) is not None:
        try:
            return float(
                metrics[key]
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

    rows = metrics.get(
        "validation_results",
        [],
    )

    if isinstance(
        rows,
        list,
    ):
        preferred_rows = [
            row
            for row in rows
            if isinstance(
                row,
                dict,
            )
            and (
                not model_name
                or row.get("model")
                == model_name
            )
            and (
                not scenario
                or row.get("scenario")
                == scenario
            )
        ]

        for row in (
            preferred_rows
            or rows
        ):
            if (
                isinstance(
                    row,
                    dict,
                )
                and row.get(key)
                is not None
            ):
                try:
                    return float(
                        row[key]
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

    return None


# ==================================================
# HTML CHỈ SỐ
# ==================================================
def metric_html(
    name: str,
    value: float | None,
    help_text: str,
) -> str:

    rendered = (
        f"{value:.4f}"
        if value is not None
        else "--"
    )

    return dedent(
        f"""
        <div class="metric-box">
            <div class="metric-name">
                {html.escape(name)}
            </div>
            <div class="metric-value">
                {rendered}
            </div>
            <div class="metric-help">
                {html.escape(help_text)}
            </div>
        </div>
        """
    ).strip()


# ==================================================
# BẢNG SO SÁNH TỪ MỘT METRICS
# ==================================================
def comparison_frame(
    metrics: dict[str, Any],
) -> pd.DataFrame:

    rows = metrics.get(
        "validation_results",
        [],
    )

    if (
        not isinstance(
            rows,
            list,
        )
        or not rows
    ):
        return pd.DataFrame()

    frame = pd.DataFrame(
        rows
    )

    if "scenario" in frame.columns:
        frame[
            "feature_set"
        ] = frame[
            "scenario"
        ].map(
            {
                "A_with_G1_G2":
                    "Có G1, G2",
                "B_without_G1_G2":
                    "Không có G1, G2",
            }
        ).fillna(
            "Không xác định"
        )

    wanted = [
        column
        for column in (
            "scenario",
            "feature_set",
            "model",
            "mae",
            "rmse",
            "r2",
        )
        if column
        in frame.columns
    ]

    frame = frame[
        wanted
    ].copy()

    frame = frame.rename(
        columns={
            "scenario":
                "Kịch bản",
            "feature_set":
                "Feature set",
            "model":
                "Mô hình",
            "mae":
                "MAE",
            "rmse":
                "RMSE",
            "r2":
                "R²",
        }
    )

    if (
        "Kịch bản"
        in frame.columns
    ):
        frame[
            "Kịch bản"
        ] = frame[
            "Kịch bản"
        ].map(
            format_scenario_short
        )

    if (
        "Mô hình"
        in frame.columns
    ):
        frame[
            "Mô hình"
        ] = frame[
            "Mô hình"
        ].map(
            format_model_name
        )

    for column in (
        "MAE",
        "RMSE",
        "R²",
    ):
        if (
            column
            in frame.columns
        ):
            frame[
                column
            ] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).round(
                4
            )

    return frame


# ==================================================
# BẢNG SO SÁNH TỔNG HỢP 2 KỊCH BẢN
# ==================================================
def build_global_comparison(
    model_items: list[dict[str, Any]],
) -> pd.DataFrame:

    frames = []

    for item in model_items:
        metrics = item.get(
            "metrics",
            {},
        )

        if not isinstance(
            metrics,
            dict,
        ):
            continue

        current_frame = (
            comparison_frame(
                metrics
            )
        )

        if not current_frame.empty:
            frames.append(
                current_frame
            )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.drop_duplicates()

    return combined


# ==================================================
# HEADER
# ==================================================
render_html(
    """
    <div class="model-hero">
        <h1>🧠 Trung tâm mô hình</h1>
        <p>
            Theo dõi hai kịch bản dự đoán G3,
            mô hình được chọn,
            hiệu năng đánh giá
            và toàn bộ đặc trưng đầu vào.
        </p>
    </div>
    """
)


# ==================================================
# THANH CÔNG CỤ
# ==================================================
toolbar_col1, toolbar_col2 = (
    st.columns(
        [1, 4]
    )
)

with toolbar_col1:

    if st.button(
        "🔄 Làm mới",
        use_container_width=True,
    ):
        load_model_info.clear()
        st.rerun()


with toolbar_col2:

    render_html(
        """
        <span class="status-ready">
            ● API model-info đang hoạt động
        </span>
        """
    )


# ==================================================
# LOAD DATA
# ==================================================
try:

    model_items = (
        load_model_info()
    )


except (
    requests.exceptions.ConnectionError
):

    st.error(
        "Không kết nối được FastAPI. "
        "Hãy chạy API tại "
        "http://127.0.0.1:8000."
    )

    st.stop()


except (
    requests.exceptions.Timeout
):

    st.error(
        "FastAPI phản hồi quá lâu."
    )

    st.stop()


except (
    requests.exceptions.HTTPError
) as exc:

    st.error(
        f"FastAPI trả về lỗi: {exc}"
    )

    st.stop()


except Exception as exc:

    st.error(
        "Không thể tải thông tin mô hình: "
        f"{exc}"
    )

    st.stop()


if not model_items:

    st.warning(
        "API chưa trả về thông tin mô hình."
    )

    st.stop()


# ==================================================
# SO SÁNH TỔNG HỢP
# ==================================================
global_comparison = (
    build_global_comparison(
        model_items
    )
)


if not global_comparison.empty:

    render_html(
        """
        <div class="section-heading">
            <h3>📊 So sánh các mô hình</h3>
            <p>
                Đối chiếu kịch bản,
                Feature set và các chỉ số
                đánh giá mô hình hồi quy.
            </p>
        </div>
        """
    )

    st.dataframe(
        global_comparison,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# TẠO TAB 2 KỊCH BẢN
# ==================================================
tab_labels = [
    (
        "🎓 Có G1, G2"
        if item.get(
            "prediction_mode"
        )
        == "with_g1_g2"
        else "🔎 Dự báo sớm"
    )
    for item
    in model_items
]


tabs = st.tabs(
    tab_labels
)


# ==================================================
# HIỂN THỊ CHI TIẾT TỪNG KỊCH BẢN
# ==================================================
for (
    tab,
    item,
) in zip(
    tabs,
    model_items,
):

    with tab:

        prediction_mode = (
            item.get(
                "prediction_mode"
            )
        )

        scenario = (
            item.get(
                "scenario"
            )
        )

        best_model = (
            item.get(
                "best_model"
            )
        )

        input_features = (
            item.get(
                "input_features",
                [],
            )
        )

        metrics = (
            item.get(
                "metrics",
                {},
            )
        )


        if not isinstance(
            input_features,
            list,
        ):
            input_features = []


        if not isinstance(
            metrics,
            dict,
        ):
            metrics = {}


        # ==========================================
        # MÔ TẢ KỊCH BẢN
        # ==========================================
        if (
            prediction_mode
            == "with_g1_g2"
        ):
            description = (
                "Kịch bản sử dụng điểm G1 và G2, "
                "phù hợp khi đã có kết quả "
                "của hai giai đoạn học trước."
            )

        else:
            description = (
                "Kịch bản hỗ trợ dự báo sớm, "
                "không cần điểm G1 và G2."
            )


        render_html(
            f"""
            <div class="scenario-banner">
                <div class="eyebrow">
                    {html.escape(
                        format_mode(
                            prediction_mode
                        )
                    )}
                </div>

                <h3>
                    {html.escape(
                        format_scenario(
                            scenario
                        )
                    )}
                </h3>

                <p>
                    {html.escape(
                        description
                    )}
                </p>
            </div>
            """
        )


        # ==========================================
        # THẺ THÔNG TIN
        # ==========================================
        (
            info_col1,
            info_col2,
            info_col3,
        ) = st.columns(
            3
        )


        with info_col1:

            render_html(
                f"""
                <div class="info-card">
                    <div class="label">
                        Mô hình tốt nhất
                    </div>

                    <div class="value">
                        {html.escape(
                            format_model_name(
                                best_model
                            )
                        )}
                    </div>

                    <div class="note">
                        Được chọn theo kết quả 
                          đánh giá của nhóm
                    </div>
                </div>
                """
            )


        with info_col2:

            render_html(
                f"""
                <div class="info-card">
                    <div class="label">
                        Số đặc trưng đầu vào
                    </div>

                    <div class="value">
                        {len(input_features)}
                    </div>

                    <div class="note">
                        Các cột model sử dụng
                        khi suy luận
                    </div>
                </div>
                """
            )


        with info_col3:

            render_html(
                f"""
                <div class="info-card">
                    <div class="label">
                        Feature set
                    </div>

                    <div class="value">
                        {html.escape(
                            format_feature_set(
                                scenario
                            )
                        )}
                    </div>

                    <div class="note">
                        Nhóm đặc trưng được sử dụng
                        trong kịch bản này
                    </div>
                </div>
                """
            )


        # ==========================================
        # CHỈ SỐ ĐÁNH GIÁ
        # ==========================================
        render_html(
            """
            <div class="section-heading">
                <h3>🎯 Chỉ số đánh giá</h3>
                <p>
                    MAE và RMSE càng thấp càng tốt;
                    R² càng gần 1 càng tốt.
                </p>
            </div>
            """
        )


        mae = get_best_metric(
            metrics,
            "mae",
            model_name=best_model,
            scenario=scenario,
        )


        rmse = get_best_metric(
            metrics,
            "rmse",
            model_name=best_model,
            scenario=scenario,
        )


        r2 = get_best_metric(
            metrics,
            "r2",
            model_name=best_model,
            scenario=scenario,
        )


        (
            metric_col1,
            metric_col2,
            metric_col3,
        ) = st.columns(
            3
        )


        with metric_col1:

            render_html(
                metric_html(
                    "MAE",
                    mae,
                    "Sai số tuyệt đối trung bình",
                )
            )


        with metric_col2:

            render_html(
                metric_html(
                    "RMSE",
                    rmse,
                    "Nhạy hơn với sai số lớn",
                )
            )


        with metric_col3:

            render_html(
                metric_html(
                    "R²",
                    r2,
                    "Mức độ giải thích biến thiên",
                )
            )


        # ==========================================
        # ĐẶC TRƯNG ĐẦU VÀO
        # ==========================================
        render_html(
            """
            <div class="section-heading">
                <h3>🧩 Đặc trưng đầu vào</h3>
                <p>
                    Các trường được truyền vào pipeline
                    khi thực hiện dự đoán.
                </p>
            </div>
            """
        )


        if input_features:

            preview_features = (
                input_features[
                    :12
                ]
            )


            chips = "".join(
                (
                    '<span class="feature-chip">'
                    f'{html.escape(str(feature))}'
                    '</span>'
                )
                for feature
                in preview_features
            )


            st.markdown(
                chips,
                unsafe_allow_html=True,
            )


            remaining = (
                len(
                    input_features
                )
                - len(
                    preview_features
                )
            )


            if remaining > 0:

                st.caption(
                    "Đang hiển thị 12 đặc trưng "
                    f"đầu tiên; còn {remaining} đặc trưng."
                )


            with st.expander(
                (
                    "Xem toàn bộ "
                    f"{len(input_features)} đặc trưng"
                ),
                expanded=False,
            ):

                feature_frame = (
                    pd.DataFrame(
                        {
                            "STT":
                                range(
                                    1,
                                    len(
                                        input_features
                                    )
                                    + 1,
                                ),
                            "Tên đặc trưng":
                                input_features,
                        }
                    )
                )


                st.dataframe(
                    feature_frame,
                    use_container_width=True,
                    hide_index=True,
                )


        else:

            st.info(
                "Chưa có danh sách đặc trưng đầu vào."
            )


        # ==========================================
        # TUNING
        # ==========================================
        tuning = metrics.get(
            "tuning"
        )


        if tuning:

            with st.expander(
                "⚙️ Chi tiết tinh chỉnh siêu tham số"
            ):

                st.json(
                    tuning
                )


        # ==========================================
        # RAW METRICS
        # ==========================================
        with st.expander(
            "🛠 Dữ liệu metrics dành cho nhóm phát triển"
        ):

            st.json(
                metrics
            )


# ==================================================
# FOOTER
# ==================================================
st.caption(
    "Thông tin được lấy trực tiếp từ FastAPI /model-info. "
    "Trang không tự đọc artifact, "
    "giúp giao diện tách biệt với tầng mô hình."
)