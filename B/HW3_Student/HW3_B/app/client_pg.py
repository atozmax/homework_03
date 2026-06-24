"""app.client_pg — Postgres read-only client (audit + source of truth)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional

import psycopg
from psycopg.rows import dict_row

from . import config


# TODO: implement ping() -> bool
# Connect to Postgres, run SELECT 1, return True if succeed, False otherwise.
# HINT: use _connect() context manager, then cur.execute("SELECT 1")

# TODO: implement _connect() context manager
# Create a psycopg connection using config.DATABASE_URL with connect_timeout=5
# Yield the connection, close in finally.
# HINT: use @contextmanager decorator

# TODO: implement fetch_corpus_hits(ids: List[str]) -> List[dict]
# Given a list of UUIDs (as strings), fetch the corresponding rows from
# the core.encoder_corpus table. Return rows in the same order as ids.
# HINT: SELECT id::text AS id, text, primary_label, labels, lang, source
# HINT: FROM core.encoder_corpus WHERE id = ANY(%s::uuid[])
# HINT: use psycopg.rows.dict_row for row factory

# TODO: (bonus) implement count_corpus() -> Optional[int]
# SELECT count(*) FROM core.encoder_corpus
