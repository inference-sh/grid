"""Retry-After parsing shared by the vendor clients: delta-seconds or an HTTP-date (RFC 9110)."""

import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def retry_delay(headers, default: float) -> float:
    """Seconds to wait from a Retry-After header, or `default` when it is absent or unusable."""
    value = headers.get("Retry-After")
    if value is None:
        return default
    try:
        delay = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return default
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delay = (when - datetime.now(timezone.utc)).total_seconds()
    return max(delay, 0.0) if math.isfinite(delay) else default
