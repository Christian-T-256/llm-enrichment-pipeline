"""Spend tracking, the hard budget cap, and the single guarded call path.

Every model call in the pipeline goes through `guarded_call`. That gives one
chokepoint for four things that otherwise get scattered (and forgotten):

1. **Hard budget cap.** A running estimate is checked after every call; the
   instant it crosses `--max-cost`, the run aborts cleanly with exit code 2.
2. **Credit-out fail-fast.** `CreditOutError` is never retried — retrying an
   empty account just burns the rate limit and the clock.
3. **Smart retry.** Only `TransientError` is retried, with exponential backoff.
   `PermanentError` (400/auth) raises immediately.
4. **Audit trail.** Per-call running totals plus an append-only spend log.

This module exists because the un-guarded version of this pipeline once retried
a credit-out error wrapped in a 5xx through an entire batch window and wasted a
real budget. The guardrails are the lesson, encoded.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime, timezone

from .config import BATCH_DISCOUNT, PRICING
from .errors import CreditOutError, PermanentError, TransientError


class SpendCapExceeded(SystemExit):
    """Raised (as SystemExit code 2) when the running estimate passes the cap."""

    def __init__(self):
        super().__init__(2)


class SpendTracker:
    """Accumulates estimated spend and enforces the cap.

    Token counts are estimates, not post-hoc actuals — the goal is a *safe upper
    bound* that trips the cap before real money is spent, which is exactly when
    you want to find out, not after reconciling an invoice.
    """

    def __init__(self, max_cost: float, log_path: str | None = None):
        self.max_cost = max_cost
        self.log_path = log_path
        self.calls = 0
        self._in: dict[str, int] = {}
        self._out: dict[str, int] = {}
        self._bin: dict[str, int] = {}   # batch input tokens (50% off)
        self._bout: dict[str, int] = {}  # batch output tokens
        self._last_log_at = 0
        self.log_every = 100

    def record(self, model: str, in_tokens: int, out_tokens: int, *, batch: bool):
        bi = self._bin if batch else self._in
        bo = self._bout if batch else self._out
        bi[model] = bi.get(model, 0) + in_tokens
        bo[model] = bo.get(model, 0) + out_tokens
        self.calls += 1
        if self.calls - self._last_log_at >= self.log_every:
            self._last_log_at = self.calls
            print(f"  [spend] ${self.estimate():.4f} across {self.calls} calls", flush=True)
        if self.estimate() > self.max_cost:
            print(f"\n[!] SPEND CAP HIT — est ${self.estimate():.4f} > cap "
                  f"${self.max_cost:.2f}. Aborting to prevent further spend.", flush=True)
            self.summary()
            self.write_log("SPEND_CAP_HIT")
            raise SpendCapExceeded()

    def estimate(self) -> float:
        total = 0.0
        for model, n in self._in.items():
            total += n * PRICING.get(model, {}).get("in", 0) / 1_000_000
        for model, n in self._out.items():
            total += n * PRICING.get(model, {}).get("out", 0) / 1_000_000
        for model, n in self._bin.items():
            total += n * PRICING.get(model, {}).get("in", 0) / 1_000_000 * BATCH_DISCOUNT
        for model, n in self._bout.items():
            total += n * PRICING.get(model, {}).get("out", 0) / 1_000_000 * BATCH_DISCOUNT
        return total

    def summary(self):
        print(f"\n  [spend] total ${self.estimate():.4f} across {self.calls} calls", flush=True)
        for model in sorted(set(self._in) | set(self._bin)):
            ri, ro = self._in.get(model, 0), self._out.get(model, 0)
            bi, bo = self._bin.get(model, 0), self._bout.get(model, 0)
            if ri or ro:
                print(f"    {model}: {ri:,} in / {ro:,} out (standard)", flush=True)
            if bi or bo:
                print(f"    {model}: {bi:,} in / {bo:,} out (batch -50%)", flush=True)

    def write_log(self, status: str = "OK"):
        if not self.log_path:
            return
        new = not os.path.exists(self.log_path)
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp", "status", "calls",
                                "estimated_cost_usd", "max_cost"])
                w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            status, self.calls, f"{self.estimate():.4f}", self.max_cost])
        except OSError as e:
            print(f"  [spend-log] could not write log: {e}", flush=True)


def guarded_call(fn, *, tracker: SpendTracker, model: str,
                 in_tokens: int = 0, out_tokens: int = 0, batch: bool = False,
                 retry: bool = True, max_retries: int = 4):
    """Run one backend call with credit-out fail-fast + transient retry, and
    record its estimated spend on success.

    `fn` is a zero-arg callable that performs the actual backend method and is
    expected to raise our `errors` taxonomy. Returns whatever `fn` returns.
    """
    backoff = 5
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = fn()
            tracker.record(model, in_tokens, out_tokens, batch=batch)
            return result
        except CreditOutError:
            raise                       # never retry; let the run abort
        except PermanentError:
            raise                       # retrying a 400 is pointless
        except TransientError as e:
            last = e
            if not retry or attempt >= max_retries:
                raise
            print(f"  [retry] {e} — attempt {attempt + 1}/{max_retries + 1}; "
                  f"backoff {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(120, backoff * 2)
    if last:
        raise last
