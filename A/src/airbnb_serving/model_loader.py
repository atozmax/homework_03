from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# TODO: import os, mlflow, mlflow.sklearn, MlflowClient when implementing.
from . import config

import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

@dataclass
class LoadedModelState:
    model: Any = None
    loaded: bool = False
    error: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


class ModelService:
    def __init__(self) -> None:
        self.state = LoadedModelState()

    def load(self) -> None:
        self.state = LoadedModelState()

        try:
            if not config.MLFLOW_TRACKING_USERNAME or not config.MLFLOW_TRACKING_PASSWORD:
                self.state.error = (
                    "MLflow credentials missing. Set MLFLOW_TRACKING_USERNAME and "
                    "MLFLOW_TRACKING_PASSWORD in C/.env or your shell environment."
                )
                return

            os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD

            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            client = MlflowClient()

            run_id = config.MLFLOW_RUN_ID or self._select_best_clean_run_id(client)
            if not run_id:
                self.state.error = (
                    "No MLflow run found. Set MLFLOW_RUN_ID or log a clean HW02 run."
                )
                return

            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            run = client.get_run(run_id)

            self.state.model = model
            self.state.loaded = True
            self.state.model_uri = model_uri
            self.state.run_id = run_id
            self.state.run_name = run.data.tags.get("mlflow.runName")
            self.state.metrics = dict(run.data.metrics)
            self.state.params = dict(run.data.params)
            self.state.tags = dict(run.data.tags)
            print('===============================================model loaded successfully===============================================')
        except Exception as exc:
            print(exc)

            self.state.loaded = False
            self.state.error = str(exc)

    def _select_best_clean_run_id(self, client: MlflowClient) -> Optional[str]:
        experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Experiment not found: {config.MLFLOW_EXPERIMENT_NAME}")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.leakage_status = 'clean' AND attributes.status = 'FINISHED'",
            order_by=["metrics.f1 DESC"],
            max_results=1,
        )
        if not runs:
            return None
        return runs[0].info.run_id

    def require_model(self):
        if not self.state.loaded or self.state.model is None:
            raise RuntimeError(self.state.error or "Model is not loaded.")
        return self.state.model

    def model_info(self) -> dict:
        return {
            "model_loaded": self.state.loaded,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
            "model_uri": self.state.model_uri,
            "run_id": self.state.run_id,
            "run_name": self.state.run_name,
            "dataset_version": config.DATASET_VERSION,
            "target": config.TARGET_NAME,
            "threshold": config.PREDICTION_THRESHOLD,
            "metrics": self.state.metrics,
            "params": self.state.params,
            "tags": self.state.tags,
            "error": self.state.error,
        }
