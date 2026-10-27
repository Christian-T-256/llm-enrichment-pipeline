"""First-run human sample-review gate.

Generation pipelines drift silently: a prompt tweak or a model bump can slowly
degrade output, and "the run completed" tells you nothing about whether the
output is any good. So on the first run for a given dataset, we dump a sample of
the post-extraction state to CSV and block until a human signals approval (by
creating a marker file). Subsequent runs of the same dataset auto-pass.

It's a deliberately low-tech gate (a CSV and a touch-file) because the point
is a forced human glance before thousands of records ship, not a UI.
"""

from __future__ import annotations

import csv
import os

SAMPLE_SIZE = 30


def _marker_path(gate_dir: str, dataset_key: str) -> str:
    return os.path.join(gate_dir, f".{dataset_key}_approved")


def write_sample(records: list[dict], gate_dir: str, dataset_key: str) -> str:
    """Write a sample of records (favoring confident segments) for review."""
    os.makedirs(gate_dir, exist_ok=True)
    ranked = sorted(records, key=lambda r: r.get("confidence", "") in ("high", "medium"),
                    reverse=True)
    sample = ranked[:SAMPLE_SIZE]
    path = os.path.join(gate_dir, f"{dataset_key}_sample_review.csv")
    cols = ["name", "company", "headline", "segment", "confidence", "source", "grounded"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sample:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def is_approved(gate_dir: str, dataset_key: str) -> bool:
    return os.path.exists(_marker_path(gate_dir, dataset_key))


def approve(gate_dir: str, dataset_key: str) -> None:
    """Create the marker (what a reviewer does to release the gate)."""
    os.makedirs(gate_dir, exist_ok=True)
    open(_marker_path(gate_dir, dataset_key), "w").close()


def require_approval(records: list[dict], gate_dir: str, dataset_key: str) -> bool:
    """Returns True if clear to proceed. On first run, writes the sample and
    returns False with instructions; the caller stops the run cleanly."""
    if is_approved(gate_dir, dataset_key):
        return True
    path = write_sample(records, gate_dir, dataset_key)
    print("\n  [review-gate] First run for this dataset: paused for review.")
    print(f"  [review-gate] Inspect the sample: {path}")
    print(f"  [review-gate] Approve to proceed:")
    print(f"      touch {_marker_path(gate_dir, dataset_key)}")
    print("  [review-gate] Then re-run the same command.\n")
    return False
