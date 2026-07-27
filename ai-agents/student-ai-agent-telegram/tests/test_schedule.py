"""Tests for pure schedule helpers."""

import datetime as dt

from app.tools import schedule
from app.tools.schedule import _miet_week_type, get_schedule_for_date


def test_miet_week_type_returns_known_value_for_cycle_start(monkeypatch):
    monkeypatch.setenv("MIET_WEEK_OFFSET", "0")
    assert _miet_week_type(dt.date(2024, 9, 1)) == 1


def test_miet_week_type_returns_known_value_for_next_week(monkeypatch):
    monkeypatch.setenv("MIET_WEEK_OFFSET", "0")
    assert _miet_week_type(dt.date(2024, 9, 8)) == 2


def test_miet_week_type_does_not_crash_on_regular_date(monkeypatch):
    monkeypatch.setenv("MIET_WEEK_OFFSET", "0")
    assert _miet_week_type(dt.date(2026, 6, 24)) in (1, 2)


class FakeResponse:
    text = '{"ok": true}'

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_get_schedule_for_date_formats_lessons_without_real_http(monkeypatch):
    today = dt.date.today()
    day_api = today.weekday() + 1
    payload = {
        "Times": [
            {"Code": 1, "TimeFrom": "2026-01-01T09:00:00", "TimeTo": "2026-01-01T10:30:00"},
            {"Code": 2, "TimeFrom": "2026-01-01T10:40:00", "TimeTo": "2026-01-01T12:10:00"},
        ],
        "Data": [
            {
                "Day": day_api,
                "Time": {"Code": 1},
                "Class": {
                    "Name": "Математика [Лек]",
                    "TeacherFull": "Иванов И.И.",
                    "Room": "101",
                },
            },
            {
                "Day": day_api,
                "Time": {"Code": 2},
                "Class": {
                    "Name": "Физика [Пр]",
                    "TeacherFull": "Петров П.П.",
                    "Room": "202",
                },
            },
        ],
    }

    def fake_post(url, data, timeout):
        assert url == "https://www.miet.ru/schedule/data"
        assert "group" in data
        assert timeout == 10
        return FakeResponse(payload)

    monkeypatch.setattr(schedule.requests, "post", fake_post)
    result = get_schedule_for_date(0)

    assert f"Расписание на {today.strftime('%d.%m.%Y')}" in result
    assert "09:00-10:30 — Математика [Лек] — Иванов И.И. — 101" in result
    assert "10:40-12:10 — Физика [Пр] — Петров П.П. — 202" in result


def test_get_schedule_for_date_handles_empty_schedule_without_real_http(monkeypatch):
    today = dt.date.today()
    payload = {
        "Times": [
            {"Code": 1, "TimeFrom": "2026-01-01T09:00:00", "TimeTo": "2026-01-01T10:30:00"},
        ],
        "Data": [],
    }

    monkeypatch.setattr(schedule.requests, "post", lambda url, data, timeout: FakeResponse(payload))
    result = get_schedule_for_date(0)

    assert f"На {today.strftime('%d.%m.%Y')}" in result
    assert "пар нет" in result
