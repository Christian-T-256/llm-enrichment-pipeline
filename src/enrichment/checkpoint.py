"""Persist upstream state right before the expensive batch submission.

The extraction stages cost real money (four model calls per record). If the
batch submission then dies (Cloudflare 5xx, a network blip, an interrupted
process), that upstream work is gone unless it was written down. This module
writes it down. `--resume-checkpoint <path>` reloads it and skips straight to
generation, turning a crash into a cheap restart instead of a full re-run.
"""

from __future__ import annotations

import csv
import os


def checkpoint_path(out_path: str) -> str:
    base = os.path.splitext(out_path)[0]
    return f"{base}__checkpoint_pre-generation.csv"


def write(records: list[dict], out_path: str) -> str:
    path = checkpoint_path(out_path)
    if not records:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  [checkpoint] wrote upstream state -> {path} ({len(records)} rows)", flush=True)
    return path


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"  [checkpoint] loaded {len(rows)} rows from {path}", flush=True)
    return rows
