from pathlib import Path

import streamlit as st


def load_css() -> None:
    """
    Tải CSS dùng chung và tạo nút chuyển đổi
    giữa giao diện sáng / tối.
    """

    # ==================================================
    # 1. ĐỌC CSS CHUNG
    # ==================================================
    css_path = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "styles.css"
    )

    if not css_path.exists():
        st.warning("Không tìm thấy file styles.css.")
    else:
        css_content = css_path.read_text(
            encoding="utf-8"
        )

        st.markdown(
            f"<style>{css_content}</style>",
            unsafe_allow_html=True,
        )

    # ==================================================
    # 2. KHỞI TẠO TRẠNG THÁI DARK MODE
    # ==================================================
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    # ==================================================
    # 3. NÚT CHUYỂN DARK MODE
    # ==================================================
    with st.sidebar:
        st.divider()

        dark_mode = st.toggle(
            "🌙 Chế độ tối",
            value=st.session_state.dark_mode,
            key="dark_mode_toggle",
        )

    st.session_state.dark_mode = dark_mode


def apply_dark_mode() -> None:
    """
    Áp dụng CSS Dark Mode.

    Hàm này phải được gọi SAU CSS riêng của từng trang
    để Dark Mode không bị CSS của trang ghi đè.
    """

    if not st.session_state.get(
        "dark_mode",
        False,
    ):
        return

    dark_css = """
    <style>

    /* ==================================================
       NỀN TOÀN BỘ ỨNG DỤNG
       ================================================== */

    html body .stApp,
    html body [data-testid="stAppViewContainer"],
    html body [data-testid="stMain"],
    html body .main,
    html body .block-container {
        background-color: #0B1120 !important;
        color: #F8FAFC !important;
    }


    /* ==================================================
       HEADER STREAMLIT
       ================================================== */

    header[data-testid="stHeader"] {
        background-color: #0B1120 !important;
    }

    [data-testid="stToolbar"] {
        background-color: transparent !important;
    }


    /* ==================================================
       SIDEBAR
       ================================================== */

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        background-color: #111827 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #E5E7EB !important;
    }

    [data-testid="stSidebarNav"] a {
        color: #CBD5E1 !important;
    }

    [data-testid="stSidebarNav"] a:hover {
        background-color: #1E293B !important;
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #334155 !important;
        color: #FFFFFF !important;
    }


    /* ==================================================
       CHỮ CHUNG
       ================================================== */

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #F8FAFC !important;
    }

    .stApp p,
    .stApp label {
        color: #CBD5E1;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #94A3B8 !important;
    }


    /* ==================================================
       SECTION TITLE
       ================================================== */

    .section-title h3,
    .section-heading h3 {
        color: #F8FAFC !important;
    }

    .section-title p,
    .section-heading p {
        color: #94A3B8 !important;
    }


    /* ==================================================
       HERO
       Giữ gradient xanh tím nhưng đảm bảo chữ sáng
       ================================================== */

    .dashboard-hero h1,
    .dashboard-hero p,
    .model-hero h1,
    .model-hero p {
        color: #FFFFFF !important;
    }


    /* ==================================================
       CARD CHUNG
       ================================================== */

    .status-card,
    .info-card,
    .metric-box,
    .prediction-result-card {
        background: #111827 !important;
        border-color: #334155 !important;
        box-shadow:
            0 8px 24px
            rgba(0, 0, 0, 0.30) !important;
    }

    .status-title,
    .info-card .label,
    .metric-box .metric-name,
    .prediction-result-label {
        color: #94A3B8 !important;
    }

    .status-value,
    .info-card .value,
    .prediction-result-score {
        color: #F8FAFC !important;
    }

    .status-note,
    .info-card .note,
    .metric-box .metric-help {
        color: #94A3B8 !important;
    }

    .metric-box .metric-value {
        color: #60A5FA !important;
    }

    .status-online {
        color: #4ADE80 !important;
    }

    .status-offline {
        color: #F87171 !important;
    }


    /* ==================================================
       MODEL INFO
       ================================================== */

    .scenario-banner {
        background:
            linear-gradient(
                135deg,
                rgba(37, 99, 235, 0.20),
                rgba(124, 58, 237, 0.14)
            ) !important;

        border-color:
            rgba(96, 165, 250, 0.30)
            !important;
    }

    .scenario-banner h3 {
        color: #F8FAFC !important;
    }

    .scenario-banner p {
        color: #CBD5E1 !important;
    }

    .scenario-banner .eyebrow {
        color: #60A5FA !important;
    }

    .feature-chip {
        background-color: #1E3A5F !important;
        color: #BFDBFE !important;
        border-color: #2563EB !important;
    }


    /* ==================================================
       FORM / CONTAINER
       ================================================== */

    [data-testid="stForm"] {
        background-color: #111827 !important;
        border-color: #334155 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: #334155 !important;
    }


    /* ==================================================
       TEXT INPUT
       ================================================== */

    [data-baseweb="input"],
    [data-baseweb="input"] > div,
    [data-baseweb="input"] input {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
        border-color: #475569 !important;
    }

    [data-baseweb="input"] input::placeholder {
        color: #94A3B8 !important;
    }


    /* ==================================================
       NUMBER INPUT
       ================================================== */

    [data-testid="stNumberInput"] input {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
    }


    /* ==================================================
       SELECTBOX
       ================================================== */

    [data-baseweb="select"] > div {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
        border-color: #475569 !important;
    }

    [data-baseweb="select"] span {
        color: #F8FAFC !important;
    }


    /* ==================================================
       TEXTAREA
       ================================================== */

    [data-baseweb="textarea"],
    [data-baseweb="textarea"] textarea {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
        border-color: #475569 !important;
    }


    /* ==================================================
       DROPDOWN
       ================================================== */

    [role="listbox"] {
        background-color: #1F2937 !important;
    }

    [role="option"] {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
    }

    [role="option"]:hover {
        background-color: #334155 !important;
    }


    /* ==================================================
       RADIO
       ================================================== */

    [data-testid="stRadio"] label {
        color: #E2E8F0 !important;
    }


    /* ==================================================
       TOGGLE
       ================================================== */

    [data-testid="stToggle"] label {
        color: #E5E7EB !important;
    }


    /* ==================================================
       EXPANDER
       ================================================== */

    [data-testid="stExpander"] {
        background-color: #111827 !important;
        border-color: #334155 !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #F8FAFC !important;
    }


    /* ==================================================
       STREAMLIT METRIC
       ================================================== */

    [data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 1rem !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }


    /* ==================================================
       DATAFRAME
       ================================================== */

    [data-testid="stDataFrame"] {
        background-color: #111827 !important;
        border-color: #334155 !important;
    }


    /* ==================================================
       TABS
       ================================================== */

    [data-baseweb="tab-list"] {
        background-color: transparent !important;
    }

    [data-baseweb="tab"] {
        color: #CBD5E1 !important;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA !important;
    }


    /* ==================================================
       BUTTON
       ================================================== */

    .stButton > button,
    .stDownloadButton > button {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border-color: #475569 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #334155 !important;
        border-color: #60A5FA !important;
    }


    /* ==================================================
       INFO / SUCCESS / WARNING
       ================================================== */

    [data-testid="stAlert"] {
        color: #E2E8F0 !important;
    }


    /* ==================================================
       GHI CHÚ TÙY CHỈNH
       ================================================== */

    .advanced-note,
    .recommend-note {
        background-color: #172033 !important;
        color: #E2E8F0 !important;
        border-color: #334155 !important;
    }


    /* ==================================================
       DIVIDER
       ================================================== */

    hr {
        border-color: #334155 !important;
    }


    /* ==================================================
       POPOVER
       ================================================== */

    [data-baseweb="popover"] {
        background-color: #111827 !important;
    }


    /* ==================================================
       SCROLLBAR
       ================================================== */

    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #0F172A;
    }

    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #64748B;
    }

    </style>
    """

    st.markdown(
        dark_css,
        unsafe_allow_html=True,
    )