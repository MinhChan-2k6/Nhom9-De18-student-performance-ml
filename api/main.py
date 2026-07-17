from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.predict import router as predict_router

app = FastAPI(
    title="Student Performance Prediction API",
    version="1.0.0",
    description="API hồi quy dự đoán điểm cuối kỳ G3.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
