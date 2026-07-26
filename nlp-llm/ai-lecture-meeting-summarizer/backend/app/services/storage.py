import json
from pathlib import Path
from typing import Any

from backend.app.config import ensure_directories, settings


def load_materials() -> dict[str, dict[str, Any]]:
    ensure_directories()
    materials_file = Path(settings.MATERIALS_FILE)
    if not materials_file.exists():
        return {}

    try:
        with materials_file.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}


def save_materials(materials: dict[str, dict[str, Any]]) -> None:
    ensure_directories()
    materials_file = Path(settings.MATERIALS_FILE)
    with materials_file.open("w", encoding="utf-8") as file:
        json.dump(materials, file, indent=2)


def add_material(metadata: dict[str, Any]) -> dict[str, Any]:
    materials = load_materials()
    materials[metadata["material_id"]] = metadata
    save_materials(materials)
    return metadata


def get_material(material_id: str) -> dict[str, Any] | None:
    materials = load_materials()
    return materials.get(material_id)
