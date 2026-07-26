"""Tests for Telegram helpers without real Telegram API calls."""

import requests

from app import telegram_bot


class FakeResponse:
    status_code = 200
    text = '{"ok": true}'

    def raise_for_status(self):
        return None


def test_send_message_posts_to_telegram(monkeypatch):
    calls = []

    def fake_post(url, data, timeout):
        calls.append((url, data, timeout))
        return FakeResponse()

    monkeypatch.setattr(telegram_bot, "get_base_url", lambda: "https://api.telegram.org/botTEST")
    monkeypatch.setattr(telegram_bot.requests, "post", fake_post)

    telegram_bot.send_message(123, "hello")

    assert calls == [
        (
            "https://api.telegram.org/botTEST/sendMessage",
            {"chat_id": 123, "text": "hello"},
            10,
        )
    ]


def test_send_message_request_error_does_not_raise(monkeypatch, caplog):
    def fake_post(url, data, timeout):
        raise requests.exceptions.ConnectionError("network down")

    monkeypatch.setattr(telegram_bot, "get_base_url", lambda: "https://api.telegram.org/botTEST")
    monkeypatch.setattr(telegram_bot.requests, "post", fake_post)

    telegram_bot.send_message(123, "hello")

    assert "Не удалось отправить сообщение в Telegram" in caplog.text
