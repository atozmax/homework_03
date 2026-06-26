"""app.config — environment variable contract for HW3_B.

Mirrors the bundle's contract. If you add an env var here, add it to:
  1. .env.example
  2. entrypoint.sh  (if it's a runtime knob, not a build-time secret)
  3. tests/test_env_contract.py  (if it affects determinism)
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- App identity ---
APP_TITLE = "QBC12 HW03-B Encoder Embedding & Search API"
APP_VERSION = "0.1.0"

# --- Bundle location (BUNDLE_DIR is the source of truth) ---
# In container: /app/bundle (baked into image)
# In dev:       ../HW3_A/bundle
BUNDLE_DIR = os.getenv("BUNDLE_DIR", "/app/bundle")
BUNDLE_DEVICE = os.getenv("BUNDLE_DEVICE", "cpu")

# --- Qdrant (shared, read-only) ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://qbc12-qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "qbc12_corpus")

# --- Postgres (shared, read-only) ---
DATABASE_HOST = os.getenv("DATABASE_HOST", "185.50.38.163")
DATABASE_PORT = os.getenv("DATABASE_PORT", "32112")
DATABASE_NAME = os.getenv("DATABASE_NAME", "qbc12_hw03_encoder")
DATABASE_API_RO_PASSWORD = os.getenv("DATABASE_API_RO_PASSWORD", "")
_raw_database_url = os.getenv("DATABASE_URL", "")
if _raw_database_url and "${" not in _raw_database_url:
    DATABASE_URL = _raw_database_url
else:
    DATABASE_URL = (
        f"postgresql://qbc12_hw03_api_ro:{DATABASE_API_RO_PASSWORD}"
        f"@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    )

# --- MLflow (for the /model-info endpoint) ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://185.50.38.163:33014")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
STUDENT_USERNAME = (
    os.getenv("STUDENT_USERNAME")
    or MLFLOW_TRACKING_USERNAME
    or "student_unknown"
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    f"qbc12_hw03_encoder_{STUDENT_USERNAME}",
)
MODEL_NAME = os.getenv("MODEL_NAME", f"qbc12_hw03_encoder_{STUDENT_USERNAME}")

# --- Search knobs ---
SEARCH_DEFAULT_TOP_K = 10
SEARCH_MAX_TOP_K = 100
SEARCH_MAX_BATCH_TEXTS = 256

# --- Embedding knobs ---
EMBED_MAX_SEQ_LEN = 256
EMBED_BATCH_HARD_CAP = 256
EMBED_DIM = 384


def get_qdrant_client(read: bool = True):
    """Return a Qdrant client (used by /predict)."""
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        timeout=10.0,
    )
