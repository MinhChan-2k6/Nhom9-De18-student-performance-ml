from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.batch_predict import router as batch_predict_router
from api.routers.health import router as health_router
from api.routers.history import router as history_router
from api.routers.model_info import router as model_info_router
from api.routers.predict import router as predict_router


app = FastAPI(
    title="Student Performance Prediction API",
    version=MODEL_VERSION,
    description="API hồi quy dự đoán điểm cuối kỳ G3.",
)


# ==================================================
# CORS
# Cho phép Streamlit gọi FastAPI
# ==================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# ROUTERS
# ==================================================
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(batch_predict_router)
app.include_router(history_router)
app.include_router(model_info_router)