"""
Regression: ZoneInfo('UTC') must return +00:00 on this host.

Root cause: /usr/share/zoneinfo/UTC contains America/Toronto data (corrupted
system tzdata). Celery's countdown→ETA conversion uses ZoneInfo('UTC'), so a
countdown=15 retry was scheduled ~4 hours late instead of 15 seconds.

Fix location: app/celery_app.py prepends the Python tzdata package to TZPATH
before importing Celery, ensuring ZoneInfo('UTC') resolves the correct file.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as dt_tz
from zoneinfo import ZoneInfo


def test_zoneinfo_utc_offset_is_zero():
    """ZoneInfo('UTC') must report a zero UTC offset after the TZPATH fix."""
    z = ZoneInfo("UTC")
    dt = datetime(2026, 6, 1, 12, 0, 0, tzinfo=z)
    offset = dt.utcoffset()
    assert offset == timedelta(0), (
        f"ZoneInfo('UTC').utcoffset() = {offset} (expected 0:00:00). "
        "System /usr/share/zoneinfo/UTC may be corrupted. "
        "The celery_app.py TZPATH fix should prevent this."
    )


def test_zoneinfo_utc_tzname_is_utc():
    """ZoneInfo('UTC') must report tzname 'UTC', not a local zone name."""
    z = ZoneInfo("UTC")
    dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=z)
    name = dt.tzname()
    assert name == "UTC", (
        f"ZoneInfo('UTC').tzname() = {name!r} (expected 'UTC'). "
        "The TZPATH fix in celery_app.py must be imported before this test runs."
    )


def test_celery_app_now_utc_offset_is_zero():
    """celery_app.now() must return a datetime with +00:00 UTC offset."""
    from app.celery_app import celery_app

    now = celery_app.now()
    assert now.utcoffset() is not None, "celery_app.now() returned naive datetime"
    assert now.utcoffset() == timedelta(0), (
        f"celery_app.now().utcoffset() = {now.utcoffset()} (expected 0:00:00). "
        f"now().isoformat() = {now.isoformat()}"
    )


def test_countdown_15_eta_is_near_future():
    """A countdown=15 ETA must be ~15s from now, not ~4h (the pre-fix behaviour)."""
    from celery.utils.time import maybe_make_aware
    from app.celery_app import celery_app

    now = celery_app.now()
    tz = celery_app.timezone
    eta = maybe_make_aware(now + timedelta(seconds=15), tz=tz)

    eta_utc = eta.astimezone(dt_tz.utc)
    wall_utc = datetime.now(dt_tz.utc)
    delta_seconds = (eta_utc - wall_utc).total_seconds()

    assert 0 < delta_seconds < 60, (
        f"countdown=15 produced an ETA {delta_seconds:.0f}s from now "
        f"(expected ~15s). ETA: {eta_utc.isoformat()}"
    )
