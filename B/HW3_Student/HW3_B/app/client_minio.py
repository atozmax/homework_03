"""app.client_minio — MinIO client (used only for bundle download in s3 mode)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from . import config


# TODO: implement get_credentials() -> dict
# Return a dict with endpoint, access_key, secret_key, bucket, prefix
# Read MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, MINIO_PREFIX from env
# HINT: use os.getenv() with sensible defaults
# HINT: prefix should include STUDENT_USERNAME, e.g. f"{config.STUDENT_USERNAME}/"


# TODO: implement download_bundle(target_dir: Path) -> bool
# Pull the bundle from MinIO into target_dir. Returns True on success.
# Only used in MODEL_SOURCE=s3 mode.
# HINT: from minio import Minio
# HINT: client = Minio(endpoint, access_key=..., secret_key=..., secure=False)
# HINT: target_dir.mkdir(parents=True, exist_ok=True)
# HINT: iterate client.list_objects(bucket, prefix=prefix, recursive=True)
# HINT: for each obj, compute rel = obj.object_name[len(prefix):], skip if not rel or obj.is_dir
# HINT: dest = target_dir / rel; dest.parent.mkdir(parents=True, exist_ok=True)
# HINT: client.fget_object(bucket, obj.object_name, str(dest))
# HINT: return True on success, False on any exception
