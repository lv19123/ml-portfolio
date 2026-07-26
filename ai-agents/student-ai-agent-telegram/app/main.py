"""Application entrypoint."""

import logging

from app.telegram_bot import run_bot


def main() -> None:
    """Run Telegram bot."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_bot()
