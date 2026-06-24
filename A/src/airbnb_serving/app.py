# Write your FastAPI app here
from dotenv import load_dotenv
from fastapi import FastAPI

from contextlib import asynccontextmanager
from airbnb_serving.schema import BatchPredictionRequest, HealthResponse, ListingFeatures, ModelInfoResponse, PredictionResponse, BatchPredictionResponse
from airbnb_serving.predictor import predict_single, predict_batch_func
from . import config
import os

from airbnb_serving.model_loader import ModelService

load_dotenv()
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID")

model_service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    print('trying to load model')
    yield


app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description="HW03 FastAPI service. Use Swagger at /docs.",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if model_service.state.loaded:
        return HealthResponse(status="ok", model_run_id=MLFLOW_RUN_ID)

    return HealthResponse(status="error", error=model_service.state.error)

@app.post("/predict")
def predict(payload: ListingFeatures):
    result =  predict_single(payload, model_service.state.model, MLFLOW_RUN_ID)
    return result

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    return predict_batch_func(payload, model_service.state.model, MLFLOW_RUN_ID)