"""hh.ru vacancies tool."""

import re

import requests
from langchain_core.tools import tool


HH_API_URL = "https://api.hh.ru/vacancies"
HH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


def _fetch_ml_jobs(count: int = 10, entry_level: bool = True) -> str:
    """Запрос к API hh.ru: вакансии по нейросетям, ML, AI."""
    params = {
        "text": "нейросети OR ML OR AI OR data science",
        "area": "1",
        "per_page": min(max(count * 3, 20), 100),
    }
    if entry_level:
        params["experience"] = "noExperience"
    items: list = []
    note = ""
    try:
        resp = requests.get(HH_API_URL, params=params, headers=HH_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
    except requests.exceptions.RequestException as e:
        err = str(e).strip() or type(e).__name__
        if getattr(e, "response", None) is not None:
            try:
                code = e.response.status_code
                err = f"HTTP {code}" + (f" — {err}" if err else "")
                print(f"[hh.ru] Ошибка: {err}", flush=True)
            except Exception:
                print(f"[hh.ru] Ошибка запроса: {e}", flush=True)
        else:
            print(f"[hh.ru] Ошибка запроса: {e}", flush=True)
        fallback = (
            "\n\nПока не получается подгрузить список с hh.ru (часто так бывает с серверов вне РФ). "
            "Открой в браузере:\n"
            "• Для новичков (без опыта), Москва: https://hh.ru/search/vacancy?text=нейросети+OR+ML+OR+AI+OR+data+science&area=1&experience=noExperience\n"
            "• Все вакансии, Москва: https://hh.ru/search/vacancy?text=нейросети+OR+ML+OR+AI+OR+data+science&area=1"
        )
        if entry_level:
            try:
                params.pop("experience", None)
                resp = requests.get(HH_API_URL, params=params, headers=HH_HEADERS, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", [])
                if items:
                    note = "\n(Показаны вакансии без фильтра «без опыта» — запрос с фильтром не прошёл.)"
                else:
                    return f"Не удалось загрузить вакансии с hh.ru: {err}{fallback}"
            except Exception:
                return f"Не удалось загрузить вакансии с hh.ru: {err}{fallback}"
        else:
            return f"Не удалось загрузить вакансии с hh.ru: {err}{fallback}"
    except Exception as e:
        err = str(e).strip() or type(e).__name__
        print(f"[hh.ru] Ошибка: {type(e).__name__} — {e}", flush=True)
        return f"Не удалось загрузить вакансии с hh.ru: {err}\n\nОткрой в браузере: https://hh.ru/search/vacancy?text=нейросети+OR+ML+OR+AI+OR+data+science&area=1"

    if not items:
        return "По заданным критериям вакансий не найдено. Попробуй без фильтра «без опыта» или изменить запрос."

    exclude_words = ("графический дизайнер", "дизайнер", "менеджер")
    items = [v for v in items if not any(w in (v.get("name") or "").lower() for w in exclude_words)]
    if not items:
        return "По заданным критериям вакансий не найдено (после отбора убраны дизайнер, менеджер). Попробуй без фильтра «без опыта» или изменить запрос."

    def _salary_key(v):
        sal = v.get("salary") or {}
        from_val = sal.get("from")
        to_val = sal.get("to")
        if from_val is not None and to_val is not None:
            return max(int(from_val), int(to_val))
        if from_val is not None:
            return int(from_val)
        if to_val is not None:
            return int(to_val)
        return 0

    items = sorted(items, key=_salary_key, reverse=True)
    lines = []
    for i, v in enumerate(items[:count], 1):
        name = (v.get("name") or "").strip()
        emp = (v.get("employer", {}) or {}).get("name", "")
        sal = v.get("salary")
        if sal:
            s_from = sal.get("from")
            s_to = sal.get("to")
            curr = (sal.get("currency") or "RUR").upper()
            if curr == "RUR":
                curr = "₽"
            try:
                if s_from is not None and s_to is not None:
                    salary_str = f"{int(s_from):,} – {int(s_to):,} {curr}".replace(",", " ")
                elif s_from is not None:
                    salary_str = f"от {int(s_from):,} {curr}".replace(",", " ")
                elif s_to is not None:
                    salary_str = f"до {int(s_to):,} {curr}".replace(",", " ")
                else:
                    salary_str = "не указана"
            except (TypeError, ValueError):
                salary_str = "не указана"
        else:
            salary_str = "не указана"
        url = (v.get("alternate_url") or "").strip()
        snippet = (v.get("snippet", {}) or {}).get("requirement") or (v.get("snippet") or {}).get("responsibility") or ""
        if snippet:
            snippet = snippet.replace("<highlighttext>", "").replace("</highlighttext>", "").strip()
            if len(snippet) > 180:
                snippet = snippet[:180] + "…"
        block = [f"{i}. {name}", f"   Работодатель: {emp}", f"   Зарплата: {salary_str}", f"   Ссылка: {url}"]
        if snippet:
            block.append(f"   Кратко: {snippet}")
        lines.append("\n".join(block))
    header = "Вакансии (отсортированы по зарплате, сначала с большей):\n\n"
    return header + "\n\n".join(lines) + note


def _parse_count(s: str, default: int = 10, max_val: int = 20) -> int:
    """Из строки вроде '10', 'штуки 2', '2 давай' извлекает число."""
    if not s or not str(s).strip():
        return default
    s = str(s).strip()
    try:
        n = int(s)
        return max(1, min(n, max_val))
    except ValueError:
        m = re.search(r"\d+", s)
        if m:
            return max(1, min(int(m.group(0)), max_val))
        return default


@tool
def get_ml_ai_jobs(count: str = "10", for_beginners: str = "да") -> str:
    """Возвращает вакансии с hh.ru: нейросети, ML, AI, data science.
    count — сколько вывести (по умолчанию 10). for_beginners — «да» или «нет»: при «да» показываются только вакансии «без опыта»."""
    n = _parse_count(count, default=10, max_val=20)
    entry = str(for_beginners).strip().lower() in ("да", "yes", "1", "true")
    return _fetch_ml_jobs(count=n, entry_level=entry)
