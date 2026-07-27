import json
import re
from pathlib import Path
from typing import Any

from backend.app.config import ensure_directories, settings
from backend.app.services.llm_client import LLMClient
from backend.app.services.processing import MaterialNotFoundError
from backend.app.services.storage import get_material, load_materials, save_materials


class StructureExtractionError(ValueError):
    pass


class StructuredOutputNotFoundError(ValueError):
    pass


TERM_STOP_WORDS = {
    "about",
    "after",
    "before",
    "lecture",
    "material",
    "notes",
    "this",
    "with",
    "и",
    "или",
    "как",
    "для",
    "что",
    "это",
    "они",
    "при",
    "над",
    "под",
    "без",
    "материал",
    "лекция",
    "тема",
}


def prepare_text_for_llm(text: str, max_chars: int = 25000) -> str:
    if len(text) <= max_chars:
        return text

    # TODO: Replace MVP truncation with map-reduce topic/term extraction.
    return (
        text[:max_chars].rstrip()
        + "\n\n[Примечание: текст был усечён для текущей MVP-версии обработки.]"
    )


def extract_json_array(text: str) -> list[dict]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Could not find a JSON array in LLM response") from None

        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Could not parse JSON array from LLM response") from exc

    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("LLM response must be a JSON array of objects")
    return parsed


def _save_material_metadata(metadata: dict[str, Any]) -> None:
    materials = load_materials()
    materials[metadata["material_id"]] = metadata
    save_materials(materials)


def _load_processed_material(material_id: str) -> dict[str, Any]:
    metadata = get_material(material_id)
    if metadata is None:
        raise MaterialNotFoundError("Material not found")
    if metadata.get("status") != "processed":
        raise StructureExtractionError(
            "Material must be processed before structure extraction"
        )
    return metadata


def _read_segments(metadata: dict[str, Any]) -> list[dict]:
    segments_path = metadata.get("segments_path")
    if not segments_path:
        raise StructureExtractionError("Processed material has no segments path")

    path = Path(segments_path)
    if not path.exists():
        raise StructureExtractionError("Segments file was not found")

    with path.open("r", encoding="utf-8") as file:
        segments = json.load(file)
    if not isinstance(segments, list) or not segments:
        raise StructureExtractionError("Segments file is empty")
    return segments


def _read_extracted_text(metadata: dict[str, Any]) -> str:
    text_path = metadata.get("extracted_text_path")
    if not text_path:
        raise StructureExtractionError("Processed material has no extracted text path")

    path = Path(text_path)
    if not path.exists():
        raise StructureExtractionError("Extracted text file was not found")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise StructureExtractionError("Extracted text is empty")
    return text


def _segments_for_prompt(segments: list[dict], max_chars: int = 8000) -> str:
    lines = []
    used_chars = 0
    for segment in segments:
        if segment.get("start") and segment.get("end"):
            source = f"{segment['start']} -> {segment['end']}"
        else:
            source = segment.get("source", "text")
        text = str(segment.get("text", "")).strip().replace("\n", " ")
        if not text:
            continue

        line = f"- {source}: {text[:500]}"
        if used_chars + len(line) > max_chars:
            break
        lines.append(line)
        used_chars += len(line)
    return "\n".join(lines)


def _compact_summary(text: str, max_length: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "..."


def _fallback_topics(segments: list[dict]) -> list[dict]:
    non_empty_segments = [
        segment for segment in segments if str(segment.get("text", "")).strip()
    ]
    selected_segments = non_empty_segments[:3] or [{"text": "Материал", "source": "text"}]

    topics = []
    for index, segment in enumerate(selected_segments, start=1):
        source_start = str(segment.get("start") or segment.get("source") or "text")
        source_end = str(segment.get("end") or segment.get("source") or source_start)
        topics.append(
            {
                "title": f"Тема {index}",
                "summary": _compact_summary(str(segment.get("text", ""))),
                "source_start": source_start,
                "source_end": source_end,
            }
        )
    return topics


def _fallback_terms(segments: list[dict]) -> list[dict]:
    terms = []
    seen = set()
    for segment in segments:
        source = str(segment.get("source") or segment.get("start") or "text")
        words = re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{4,}", str(segment.get("text", "")))
        for word in words:
            normalized = word.lower()
            if normalized in seen or normalized in TERM_STOP_WORDS:
                continue
            seen.add(normalized)
            terms.append(
                {
                    "term": word,
                    "definition": f"Термин встречается в материале; уточните значение по источнику {source}.",
                    "source": source,
                }
            )
            if len(terms) >= 5:
                return terms

    if not terms:
        terms.append(
            {
                "term": "Материал",
                "definition": "Общее понятие из загруженного текста.",
                "source": "text",
            }
        )
    return terms


def _validate_topics(topics: list[dict]) -> list[dict]:
    validated = []
    for topic in topics:
        validated.append(
            {
                "title": str(topic.get("title", "")).strip(),
                "summary": str(topic.get("summary", "")).strip(),
                "source_start": str(topic.get("source_start", "")).strip() or "text",
                "source_end": str(topic.get("source_end", "")).strip() or "text",
            }
        )
    validated = [topic for topic in validated if topic["title"] and topic["summary"]]
    if not validated:
        raise StructureExtractionError("No topics were extracted")
    return validated


def _validate_terms(terms: list[dict]) -> list[dict]:
    validated = []
    for term in terms:
        validated.append(
            {
                "term": str(term.get("term", "")).strip(),
                "definition": str(term.get("definition", "")).strip(),
                "source": str(term.get("source", "")).strip() or "text",
            }
        )
    validated = [term for term in validated if term["term"] and term["definition"]]
    if not validated:
        raise StructureExtractionError("No key terms were extracted")
    return validated


def _save_topics(metadata: dict[str, Any], topics: list[dict]) -> dict:
    processed_dir = Path(settings.PROCESSED_DIR) / metadata["material_id"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    topics_path = processed_dir / "topics.json"

    with topics_path.open("w", encoding="utf-8") as file:
        json.dump(topics, file, indent=2, ensure_ascii=False)

    metadata["topics_path"] = str(topics_path)
    metadata["topics_count"] = len(topics)
    _save_material_metadata(metadata)
    return {
        "material_id": metadata["material_id"],
        "status": "created",
        "topics_count": len(topics),
        "topics": topics,
    }


def _save_terms(metadata: dict[str, Any], terms: list[dict]) -> dict:
    processed_dir = Path(settings.PROCESSED_DIR) / metadata["material_id"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    terms_path = processed_dir / "terms.json"

    with terms_path.open("w", encoding="utf-8") as file:
        json.dump(terms, file, indent=2, ensure_ascii=False)

    metadata["terms_path"] = str(terms_path)
    metadata["terms_count"] = len(terms)
    _save_material_metadata(metadata)
    return {
        "material_id": metadata["material_id"],
        "status": "created",
        "terms_count": len(terms),
        "terms": terms,
    }


def generate_topics(material_id: str) -> dict:
    metadata = _load_processed_material(material_id)
    segments = _read_segments(metadata)
    text = prepare_text_for_llm(_read_extracted_text(metadata))

    if not (settings.OPENROUTER_API_KEY or "").strip():
        topics = _fallback_topics(segments)
        return _save_topics(metadata, topics)

    prompt = f"""
Ты извлекаешь темы из учебного или встречного материала.
Верни только валидный JSON-массив, без Markdown и пояснений.

Формат:
[
  {{
    "title": "...",
    "summary": "...",
    "source_start": "...",
    "source_end": "..."
  }}
]

Правила:
- Не выдумывай факты.
- Используй только предоставленный материал.
- Создай 3-10 тем в зависимости от размера текста.
- Сохраняй исходный порядок материала.
- Используй ссылки на источники из сегментов, где возможно.
- Если границы темы неясны, используй ближайший доступный source.

Сегменты и источники:
{_segments_for_prompt(segments)}

Текст:
\"\"\"
{text}
\"\"\"
""".strip()
    topics = _validate_topics(extract_json_array(LLMClient().generate(prompt)))
    return _save_topics(metadata, topics)


def generate_key_terms(material_id: str) -> dict:
    metadata = _load_processed_material(material_id)
    segments = _read_segments(metadata)
    text = prepare_text_for_llm(_read_extracted_text(metadata))

    if not (settings.OPENROUTER_API_KEY or "").strip():
        terms = _fallback_terms(segments)
        return _save_terms(metadata, terms)

    prompt = f"""
Ты извлекаешь ключевые термины из учебного или встречного материала.
Верни только валидный JSON-массив, без Markdown и пояснений.

Формат:
[
  {{
    "term": "...",
    "definition": "...",
    "source": "..."
  }}
]

Правила:
- Извлекай важные понятия, определения, модели, методы, имена или технические термины.
- Не выдумывай термины, которых нет в материале.
- Определения должны быть короткими и ясными.
- Используй ссылки на источники из сегментов, где возможно.
- Верни 5-20 терминов в зависимости от размера текста.

Сегменты и источники:
{_segments_for_prompt(segments)}

Текст:
\"\"\"
{text}
\"\"\"
""".strip()
    terms = _validate_terms(extract_json_array(LLMClient().generate(prompt)))
    return _save_terms(metadata, terms)


def _read_structured_output(
    material_id: str,
    path_key: str,
    missing_message: str,
) -> list[dict]:
    metadata = get_material(material_id)
    if metadata is None:
        raise MaterialNotFoundError("Material not found")

    output_path = metadata.get(path_key)
    if not output_path:
        raise StructuredOutputNotFoundError(missing_message)

    path = Path(output_path)
    if not path.exists():
        raise StructuredOutputNotFoundError(missing_message)

    with path.open("r", encoding="utf-8") as file:
        output = json.load(file)
    if not isinstance(output, list):
        raise StructureExtractionError("Structured output file is invalid")
    return output


def get_topics(material_id: str) -> list[dict]:
    return _read_structured_output(
        material_id,
        "topics_path",
        "Topics have not been generated yet",
    )


def get_key_terms(material_id: str) -> list[dict]:
    return _read_structured_output(
        material_id,
        "terms_path",
        "Key terms have not been generated yet",
    )
