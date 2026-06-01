"""Backend-agnostic error taxonomy.

The pipeline never catches vendor SDK exceptions directly. Each backend maps its
own failures onto these three classes, so the retry/guard logic stays identical
whether it's talking to the real API or the mock. The split that matters is
*permanent vs transient vs out-of-money* — those three demand different
responses, and conflating them is how you burn a budget retrying a 400.
"""

from __future__ import annotations


class CreditOutError(Exception):
    """Account is out of credit / over a hard billing limit.

    Permanent and special: never retried, and it aborts the whole run rather
    than failing a single record. Critically, some endpoints (notably Batch)
    wrap this in a 5xx, so backends must sniff the message body, not just the
    HTTP status, before classifying.
    """


class PermanentError(Exception):
    """A request that will fail identically if retried (400, auth, not-found).

    Raised immediately. Retrying these only wastes time and, on metered
    endpoints, money.
    """


class TransientError(Exception):
    """A request that might succeed on retry (5xx, connection reset, rate limit)."""
