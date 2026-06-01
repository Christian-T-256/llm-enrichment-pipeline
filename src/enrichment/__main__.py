"""CLI entrypoint.

    python -m enrichment --input examples/sample_records.csv \
        --output .runs/out.csv --max-cost 5 --dry-run

`--max-cost` is mandatory by design: there is no way to start a live run without
declaring a ceiling. `--dry-run` uses the offline mock backend (no key, no
spend) and still prints a projected cost from the same estimator.
"""

from __future__ import annotations

import argparse
import os
import sys

from .backends import AnthropicBackend, MockBackend
from .config import Settings
from .context import RunContext
from .errors import CreditOutError
from .pipeline import run
from .review_gate import approve
from .spend import SpendTracker


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _dataset_key(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="enrichment", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="input CSV of records")
    p.add_argument("--output", help="output CSV path")
    p.add_argument("--max-cost", type=float,
                   help="hard USD spend cap (REQUIRED for any run)")
    p.add_argument("--dry-run", action="store_true",
                   help="use the offline mock backend (no API key, no spend)")
    p.add_argument("--resume-checkpoint", metavar="PATH",
                   help="resume from a pre-generation checkpoint CSV")
    p.add_argument("--dataset-key", help="review-gate key (default: input filename)")
    p.add_argument("--gate-dir", default="./review_gate",
                   help="directory for review-gate samples + markers")
    p.add_argument("--no-review-gate", action="store_true",
                   help="skip the first-run human review gate")
    p.add_argument("--spend-log", default="./spend_log.csv",
                   help="append-only spend audit log path")
    p.add_argument("--approve", metavar="DATASET_KEY",
                   help="approve a dataset's review gate, then exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.approve:
        approve(args.gate_dir, args.approve)
        print(f"Approved review gate for '{args.approve}'.")
        return 0

    if not (args.input and args.output):
        print("error: --input and --output are required.", file=sys.stderr)
        return 64
    if args.max_cost is None:
        print("error: --max-cost is required. Declare a spend ceiling before "
              "running (use a small value with --dry-run to estimate).",
              file=sys.stderr)
        return 64

    settings = Settings(
        max_cost=args.max_cost,
        dry_run=args.dry_run,
        resume_checkpoint=args.resume_checkpoint,
        review_gate=not args.no_review_gate,
    )

    if args.dry_run:
        backend = MockBackend()
        print("[mode] DRY RUN — offline mock backend, no API key, no spend.")
    else:
        _load_dotenv()
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("error: ANTHROPIC_API_KEY not set (use --dry-run to run offline).",
                  file=sys.stderr)
            return 64
        backend = AnthropicBackend(key)

    tracker = SpendTracker(max_cost=args.max_cost, log_path=args.spend_log)
    ctx = RunContext(backend=backend, tracker=tracker, settings=settings)
    dataset_key = args.dataset_key or _dataset_key(args.input)

    try:
        summary = run(args.input, args.output, ctx,
                      dataset_key=dataset_key, gate_dir=args.gate_dir)
    except CreditOutError as e:
        print(f"\n[!] CREDIT-OUT — aborting without retry: {e}", file=sys.stderr)
        tracker.write_log("CREDIT_OUT")
        return 3

    print("\n" + "=" * 56)
    for k, v in summary.items():
        print(f"  {k:>20}: {v}")
    print("=" * 56)
    return 0 if summary.get("status") in ("OK", "AWAITING_REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
