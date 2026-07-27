"""Profi.ru links tool."""

from langchain_core.tools import tool


def _profi_orders_instruction(prefix: str = "") -> str:
    """Краткая инструкция со ссылкой на заказы Профи.ру."""
    return (
        f"{prefix}"
        "Заказы Профи.ру (IT-фриланс)\n\n"
        "Войти в кабинет и смотреть заказы:\n"
        "https://profi.ru/backoffice/n.php\n\n"
        "Раздел IT-фриланс (вход с главной):\n"
        "https://profi.ru/rabota/it_freelance/"
    )


def _fetch_profi_orders(count: int = 10) -> str:
    """Возвращает аккуратную инструкцию со ссылкой на заказы Профи.ру."""
    return _profi_orders_instruction()


@tool
def get_profi_orders(count: str = "10") -> str:
    """Профи.ру: возвращает готовый текст со ссылками на кабинет и раздел IT-фриланс. Выводи результат целиком."""
    return _fetch_profi_orders()
