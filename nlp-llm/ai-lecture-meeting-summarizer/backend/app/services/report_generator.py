from pathlib import Path
from typing import Any

from backend.app.config import ensure_directories, settings
from backend.app.services.llm_client import LLMClient
from backend.app.services.processing import MaterialNotFoundError
from backend.app.services.storage import get_material, load_materials, save_materials


class ReportGenerationError(ValueError):
    pass


class ReportNotFoundError(ValueError):
    pass


VALID_REPORT_TYPES = {"short": "short", "full-clean": "full_clean"}


def prepare_text_for_llm(text: str, max_chars: int = 25000) -> str:
    if len(text) <= max_chars:
        return text

    # TODO: Replace MVP truncation with map-reduce summarization for long materials.
    return (
        text[:max_chars].rstrip()
        + "\n\n[Примечание: текст был усечён для текущей MVP-версии обработки.]"
    )


def _save_material_metadata(metadata: dict[str, Any]) -> None:
    materials = load_materials()
    materials[metadata["material_id"]] = metadata
    save_materials(materials)


def _load_processed_material(material_id: str) -> dict[str, Any]:
    metadata = get_material(material_id)
    if metadata is None:
        raise MaterialNotFoundError("Material not found")
    if metadata.get("status") != "processed":
        raise ReportGenerationError("Material must be processed before report generation")
    if not metadata.get("extracted_text_path"):
        raise ReportGenerationError("Processed material has no extracted text path")
    return metadata


def _read_extracted_text(metadata: dict[str, Any]) -> str:
    text_path = Path(metadata["extracted_text_path"])
    if not text_path.exists():
        raise ReportGenerationError("Extracted text file was not found")

    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ReportGenerationError("Extracted text is empty")
    return text


def _save_report(
    metadata: dict[str, Any],
    report_key: str,
    filename: str,
    content: str,
) -> dict[str, str]:
    reports_dir = Path(settings.REPORTS_DIR) / metadata["material_id"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / filename
    report_path.write_text(content, encoding="utf-8")

    reports = metadata.get("reports") or {}
    reports[report_key] = {"path": str(report_path), "format": "markdown"}
    metadata["reports"] = reports
    _save_material_metadata(metadata)

    return {
        "material_id": metadata["material_id"],
        "report_type": report_key,
        "status": "created",
        "report_path": str(report_path),
    }


def _generate_report(material_id: str, report_key: str, filename: str, prompt: str) -> dict:
    metadata = _load_processed_material(material_id)
    ensure_directories()
    markdown = LLMClient().generate(prompt).strip()
    if not markdown:
        raise ReportGenerationError("Generated report is empty")
    return _save_report(metadata, report_key, filename, markdown)


def generate_short_report(material_id: str) -> dict:
    metadata = _load_processed_material(material_id)
    text = prepare_text_for_llm(_read_extracted_text(metadata))
    prompt = f"""
Сгенерируй компактный структурированный отчёт на русском языке.
Используй только предоставленный текст. Не выдумывай факты.
Не переписывай всю лекцию целиком.

Формат Markdown:

# Краткий отчёт

## О чём материал

## Основные темы

## Краткое содержание

## Что важно запомнить

Текст материала:
\"\"\"
{text}
\"\"\"
""".strip()
    return _generate_report(material_id, "short", "short_report.md", prompt)


def generate_full_clean_notes(material_id: str) -> dict:
    metadata = _load_processed_material(material_id)
    text = prepare_text_for_llm(_read_extracted_text(metadata))
    prompt = f"""
Преврати извлечённый материал в полный очищенный конспект на русском языке.
Это НЕ краткое резюме: сохрани содержание лекции настолько полно, насколько разумно.
Сохраняй исходный порядок объяснения.
Убери слова-паразиты, повторы, сбитую речь и бесполезные фрагменты.
Сделай текст читаемым и хорошо структурированным.
Не выдумывай факты.
Если в будущем появятся страницы или таймкоды, для них можно оставить естественные места в структуре, но сейчас используй обычные заголовки.

Формат Markdown должен начинаться так:

# Полный очищенный конспект

Затем добавь разделы с понятными заголовками.

Текст материала:
\"\"\"
{text}
\"\"\"
""".strip()
    return _generate_report(
        material_id,
        "full_clean",
        "full_clean_notes.md",
        prompt,
    )


def normalize_report_type(report_type: str) -> str:
    if report_type not in VALID_REPORT_TYPES:
        raise ReportGenerationError("Invalid report_type")
    return VALID_REPORT_TYPES[report_type]


def get_report_path(material_id: str, report_type: str) -> Path:
    metadata = get_material(material_id)
    if metadata is None:
        raise MaterialNotFoundError("Material not found")

    report_key = normalize_report_type(report_type)
    report = (metadata.get("reports") or {}).get(report_key)
    if not report or not report.get("path"):
        raise ReportNotFoundError("Report not found")

    report_path = Path(report["path"])
    if not report_path.exists():
        raise ReportNotFoundError("Report not found")
    return report_path


def read_report(material_id: str, report_type: str) -> dict:
    report_path = get_report_path(material_id, report_type)
    return {
        "material_id": material_id,
        "report_type": report_type,
        "content": report_path.read_text(encoding="utf-8"),
    }
