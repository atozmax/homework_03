from __future__ import annotations

from typing import Iterable, List

import pandas as pd
import numpy as np

from fastapi import HTTPException, status

from . import config
from .schema import BatchPredictionRequest, BatchPredictionResponse, ListingFeatures, PredictionResponse


def records_to_dataframe(records: Iterable[ListingFeatures]) -> pd.DataFrame:
    """Convert validated API payloads into the exact DataFrame expected by the model."""
    rows = [record.model_dump() for record in records]
    df = pd.DataFrame(rows)

    for c in df.columns:
        if c in config.FORBIDDEN_FIELDS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "field not expected", "unknown_fields": [c]},
            )

    missing_cols = [c for c in config.EXPECTED_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "missing feature handling", "missing_fields": missing_cols},
        )

    return df[config.EXPECTED_FEATURE_COLUMNS].rename(columns=config.COLUMNS_TO_RENAME)


def predict_single(payload: ListingFeatures, model, run_id: str) -> PredictionResponse:
    X = records_to_dataframe([payload])

    probability = model.predict_proba(X)[:, 1]
    prediction = (probability > config.PREDICTION_THRESHOLD).astype(int)
    response = PredictionResponse(
            prediction=prediction[0],
            probability_high_demand=probability[0],
            model_run_id=config.MLFLOW_RUN_ID,  
        )
    
    return  response

def predict_batch_func(payload: BatchPredictionRequest, model, run_id: str) -> BatchPredictionResponse:
    X = records_to_dataframe(payload.records)
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability > config.PREDICTION_THRESHOLD).astype(int)
    response = BatchPredictionResponse(
        count=len(payload.records),
        predictions=[PredictionResponse(prediction=prediction[i], probability_high_demand=probability[i], model_run_id=run_id) for i in range(len(payload.records))],
    )
    return response