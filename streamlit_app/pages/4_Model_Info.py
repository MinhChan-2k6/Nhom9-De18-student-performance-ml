from __future__ import annotations

import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Thông tin mô hình",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Thông tin mô hình")
st.caption("Thông tin và chỉ số đánh giá các mô hình dự đoán G3.")


@st.cache_data(ttl=60)
def load_model_info() -> list[dict]:
    response = requests.get(
        f"{API_BASE_URL}/model-info",
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


try:
    model_list = load_model_info()

    if not model_list:
        st.warning("Chưa có thông tin mô hình.")
        st.stop()

    for index, item in enumerate(model_list, start=1):
        prediction_mode = item.get("prediction_mode", "Không xác định")
        scenario = item.get("scenario", "Không xác định")
        best_model = item.get("best_model", "Không xác định")
        input_features = item.get("input_features", [])
        metrics = item.get("metrics", {})

        st.subheader(f"Mô hình {index}: {scenario}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Chế độ dự đoán",
            prediction_mode,
        )
        col2.metric(
            "Mô hình tốt nhất",
            best_model,
        )
        col3.metric(
            "Số đặc trưng đầu vào",
            len(input_features),
        )

        if input_features:
            with st.expander("Danh sách đặc trưng đầu vào"):
                st.write(", ".join(input_features))

        with st.expander("Chi tiết metrics và tuning"):
            st.json(metrics)

        st.divider()

except requests.exceptions.ConnectionError:
    st.error(
        "Không kết nối được FastAPI. "
        "Hãy chạy API tại http://127.0.0.1:8000."
    )

except requests.exceptions.Timeout:
    st.error("API phản hồi quá lâu.")

except requests.exceptions.HTTPError as exc:
    st.error(f"API trả về lỗi: {exc}")

except Exception as exc:
    st.error(f"Không thể tải thông tin mô hình: {exc}")