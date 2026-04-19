from __future__ import annotations

from app.services.session_stream_service import _prepare_initial_log_batch


def test_prepare_initial_log_batch_orders_oldest_to_newest_and_tracks_max_id():
    recent_logs = [
        {"id": 12, "message": "latest"},
        {"id": 9, "message": "middle"},
        {"id": 4, "message": "oldest"},
    ]

    ordered, last_log_id = _prepare_initial_log_batch(recent_logs)

    assert [entry["id"] for entry in ordered] == [4, 9, 12]
    assert last_log_id == 12
