"""MIET schedule tool."""

import datetime as dt
import os

import requests
from langchain_core.tools import tool

from app.config import GROUP


def _miet_week_type(target: dt.date) -> int:
    """Возвращает тип недели для фильтра API: 1 = числитель, 2 = знаменатель."""
    start = dt.date(target.year, 9, 1)
    if target < start:
        start = dt.date(target.year - 1, 9, 1)
    week_index = (target - start).days // 7
    offset = int(os.getenv("MIET_WEEK_OFFSET", "0"))
    cycle = (week_index + offset) % 4
    return 1 if cycle in (0, 2) else 2


def get_schedule_for_date(day_offset: int) -> str:
    """Возвращает расписание на заданный день (0=сегодня, 1=завтра)."""
    target = dt.date.today() + dt.timedelta(days=day_offset)

    url = "https://www.miet.ru/schedule/data"
    data = {
        "group": GROUP,
    }

    try:
        resp = requests.post(url, data=data, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Не удалось получить расписание для {GROUP}: {e}"

    try:
        data_json = resp.json()
    except Exception as e:
        return (
            f"Сервер вернул не JSON. Ошибка парсинга: {e}\n\n"
            f"Фрагмент ответа:\n{resp.text[:500]}"
        )

    times_list = data_json.get("Times", [])
    data_list = data_json.get("Data", [])

    times_by_code = {t.get("Code"): t for t in times_list}
    week_type = _miet_week_type(target)

    weekday_py = target.weekday()
    day_api = weekday_py + 1
    all_lessons = [row for row in data_list if row.get("Day") == day_api]

    def _base_subject(name: str) -> str:
        for suffix in (" [Пр]", " [Лаб]", " [Лек]"):
            if suffix in name:
                return name.split(suffix)[0].strip()
        return name

    has_week_field = any((row.get("Week") or row.get("WeekType")) is not None for row in all_lessons)
    if has_week_field:
        lessons = [row for row in all_lessons if int(row.get("Week") or row.get("WeekType")) == week_type]
    else:
        by_base: dict[str, list[dict]] = {}
        for row in all_lessons:
            name = (row.get("Class") or {}).get("Name", "") or ""
            base = _base_subject(name)
            by_base.setdefault(base, []).append(row)
        has_both_pro_lab = {
            base: (any("[Пр]" in ((r.get("Class") or {}).get("Name", "") or "") for r in rows)
                   and any("[Лаб]" in ((r.get("Class") or {}).get("Name", "") or "") for r in rows))
            for base, rows in by_base.items()
        }
        chosen = []
        for row in all_lessons:
            base = _base_subject((row.get("Class") or {}).get("Name", "") or "")
            name = (row.get("Class") or {}).get("Name", "") or ""
            if not has_both_pro_lab.get(base, False):
                chosen.append(row)
                continue
            if week_type == 1 and "[Лаб]" not in name:
                continue
            if week_type == 2 and "[Пр]" not in name:
                continue
            chosen.append(row)
        lessons = sorted(chosen, key=lambda r: r.get("Time", {}).get("Code", 0))

    if not lessons:
        return f"На {target.strftime('%d.%m.%Y')} у группы {GROUP} пар нет по данным расписания (неделя: {'числитель' if week_type == 1 else 'знаменатель'})."

    lessons.sort(key=lambda r: r.get("Time", {}).get("Code", 0))

    seen = set()
    slot_to_lines: dict[int, list[str]] = {}
    for row in lessons:
        time_info = row.get("Time", {})
        code = time_info.get("Code")
        t_meta = times_by_code.get(code, {})
        t_from = t_meta.get("TimeFrom", "")
        t_to = t_meta.get("TimeTo", "")
        t_from_str = t_from[11:16] if len(t_from) >= 16 else ""
        t_to_str = t_to[11:16] if len(t_to) >= 16 else ""
        cls = row.get("Class", {}) or {}
        subj = cls.get("Name", "Без названия")
        teacher = cls.get("TeacherFull") or cls.get("Teacher") or ""
        room = cls.get("Room") or cls.get("Auditory") or ""
        key = (t_from_str, t_to_str, subj, teacher)
        if key in seen:
            continue
        seen.add(key)
        parts = [f"{t_from_str}-{t_to_str}", subj]
        if teacher:
            parts.append(teacher)
        if room:
            parts.append(room)
        line = " — ".join(parts)
        slot_to_lines.setdefault(code, []).append(line)

    codes_used = sorted(slot_to_lines.keys())
    min_code, max_code = min(codes_used), max(codes_used)
    all_codes_sorted = sorted(times_by_code.keys())
    codes_in_range = [c for c in all_codes_sorted if min_code <= c <= max_code]

    lines = []
    for code in codes_in_range:
        t_meta = times_by_code.get(code, {})
        t_from = t_meta.get("TimeFrom", "")
        t_to = t_meta.get("TimeTo", "")
        t_from_str = t_from[11:16] if len(t_from) >= 16 else ""
        t_to_str = t_to[11:16] if len(t_to) >= 16 else ""
        if code in slot_to_lines:
            for ln in slot_to_lines[code]:
                lines.append(ln)
        else:
            lines.append(f"{t_from_str}-{t_to_str} — окно")

    body = "\n".join(lines)
    week_label = "числитель" if week_type == 1 else "знаменатель"
    return f"Расписание на {target.strftime('%d.%m.%Y')} для группы {GROUP} ({week_label}):\n\n{body}"


@tool
def get_schedule_tool(day: str = "завтра") -> str:
    """Возвращает расписание по данным МИЭТ для группы студента.
    day — «сегодня» или «завтра» (по умолчанию завтра)."""
    s = (day or "завтра").strip().lower()
    if s in ("сегодня", "today"):
        return get_schedule_for_date(0)
    return get_schedule_for_date(1)


@tool
def get_tomorrow_schedule_tool(_: str = "") -> str:
    """Возвращает расписание на завтра по данным МИЭТ для группы студента."""
    return get_schedule_for_date(1)
