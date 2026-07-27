"""Telegram long polling loop."""

import logging
import time

import requests

from app.agent_core import ask_ai
from app.config import get_base_url, require_bot_token
from app.rag import _get_rag_retriever


logger = logging.getLogger(__name__)


def send_message(chat_id: int, text: str) -> None:
    """Отправка текста в чат через Telegram Bot API."""
    base_url = get_base_url()
    url = f"{base_url}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        response = getattr(e, "response", None)
        if response is not None:
            logger.warning("Telegram API вернул ошибку при отправке сообщения: HTTP %s — %s", response.status_code, response.text[:500])
        else:
            logger.warning("Не удалось отправить сообщение в Telegram: %s", e)
    except Exception:
        logger.exception("Непредвиденная ошибка при отправке сообщения в Telegram")


def delete_webhook() -> bool:
    """Сбрасывает webhook, чтобы getUpdates (long polling) работал."""
    try:
        base_url = get_base_url()
        resp = requests.get(f"{base_url}/deleteWebhook", params={"drop_pending_updates": True}, timeout=10)
        resp.raise_for_status()
        logger.info("Webhook сброшен, запускаю приём сообщений...")
        return True
    except Exception as e:
        logger.warning("Ошибка deleteWebhook: %s", e)
        return False


def run_bot() -> None:
    """Запускает Telegram long polling loop."""
    require_bot_token()
    delete_webhook()
    time.sleep(2)

    rag_ready = _get_rag_retriever() is not None
    if rag_ready:
        logger.info("Лекции подгружены, всё ок.")
    else:
        logger.info("Лекции не загружены (папка materials пуста или ошибка). Бот работает без базы материалов.")

    logger.info("Бот запущен. Пиши боту что угодно — он сам выберет инструмент (расписание, калькулятор, время, вакансии, заказы Профи).")

    offset = None
    conflict_retries = 0

    while True:
        url = f"{get_base_url()}/getUpdates"
        params = {"timeout": 10}
        if offset is not None:
            params["offset"] = offset

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            conflict_retries = 0
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                conflict_retries += 1
                logger.warning("409 Conflict (другой процесс или webhook). Попытка %s: сбрасываю webhook и жду 5 сек...", conflict_retries)
                delete_webhook()
                time.sleep(5)
            else:
                logger.warning("Ошибка getUpdates: %s", e)
                time.sleep(3)
            continue
        except Exception as e:
            logger.warning("Ошибка getUpdates: %s", e)
            time.sleep(3)
            continue

        for update in data.get("result", []):
            offset = update["update_id"] + 1

            message = update.get("message")
            if not message:
                continue

            chat = message.get("chat", {})
            chat_id = chat.get("id")
            text = (message.get("text") or "").strip()

            if not text or chat_id is None:
                continue

            reply = ask_ai(chat_id, text)
            send_message(chat_id, reply)

        time.sleep(1)
