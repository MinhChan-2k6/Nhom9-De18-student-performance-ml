from pathlib import Path

import streamlit as st


def load_css() -> None:
    """Đọc file CSS dùng chung và áp dụng cho giao diện."""

    css_path = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "styles.css"
    )

    if not css_path.exists():
        st.warning("Không tìm thấy file styles.css.")
        return

    css_content = css_path.read_text(
        encoding="utf-8"
    )

    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True,
    )