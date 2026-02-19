"""Temporal and session-based features."""

from __future__ import annotations

from datetime import datetime

from fxer.config.constants import (
    ASIAN_SESSION,
    LONDON_SESSION,
    NY_SESSION,
    OVERLAP_SESSION,
)


def is_london_session(ts: datetime) -> bool:
    """Check if timestamp falls within London session (07:00-16:00 UTC)."""
    return LONDON_SESSION.is_active(ts.hour)


def is_ny_session(ts: datetime) -> bool:
    """Check if timestamp falls within New York session (12:00-21:00 UTC)."""
    return NY_SESSION.is_active(ts.hour)


def is_overlap_session(ts: datetime) -> bool:
    """Check if timestamp falls within London/NY overlap (12:00-16:00 UTC)."""
    return OVERLAP_SESSION.is_active(ts.hour)


def is_asian_session(ts: datetime) -> bool:
    """Check if timestamp falls within Asian session (22:00-07:00 UTC)."""
    return ASIAN_SESSION.is_active(ts.hour)


def hour_of_day(ts: datetime) -> int:
    """Return UTC hour (0-23)."""
    return ts.hour


def day_of_week(ts: datetime) -> int:
    """Return day of week (0=Monday, 6=Sunday)."""
    return ts.weekday()


def is_month_turn(ts: datetime) -> bool:
    """Check if timestamp is within first or last 2 trading days of the month.

    Uses calendar days 1-2 for start and last 2 calendar days for end
    as a practical approximation.
    """
    day = ts.day
    # First 2 days of month
    if day <= 2:
        return True
    # Last 2 days of month: check by advancing to next month
    import calendar

    _, last_day = calendar.monthrange(ts.year, ts.month)
    if day >= last_day - 1:
        return True
    return False


def compute_time_features(ts: datetime) -> dict[str, bool | int]:
    """Compute all temporal features for a timestamp.

    Returns a dict with keys matching FeatureVector field names.
    """
    return {
        "is_london_session": is_london_session(ts),
        "is_ny_session": is_ny_session(ts),
        "is_overlap_session": is_overlap_session(ts),
        "is_asian_session": is_asian_session(ts),
        "hour_of_day": hour_of_day(ts),
        "day_of_week": day_of_week(ts),
        "is_month_turn": is_month_turn(ts),
    }
