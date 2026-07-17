import streamlit as st

# Cấu hình chung cho ứng dụng
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Trang giới thiệu
st.title("🎓 Student Performance Prediction")

st.write(
    "Hệ thống sử dụng Machine Learning để dự đoán "
    "điểm cuối kỳ G3 của học sinh."
)

st.info(
    "Chọn một chức năng trong thanh menu bên trái để bắt đầu."
)