import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.config import ensure_directories, settings
from backend.app.services.chunker import chunk_segments
from backend.app.services.llm_client import LLMClient
from backend.app.services.processing import MaterialNotFoundError
from backend.app.services.storage import get_material, load_materials, save_materials


class RagError(ValueError):
    pass


def _save_material_metadata(metadata: dict[str, Any]) -> None:
    materials = load_materials()
    materials[metadata["material_id"]] = metadata
    save_materials(materials)


def _load_material(material_id: str) -> dict[str, Any]:
    metadata = get_material(material_id)
    if metadata is None:
        raise MaterialNotFoundError("Material not found")
    return metadata


def _load_processed_material(material_id: str) -> dict[str, Any]:
    metadata = _load_material(material_id)
    if metadata.get("status") != "processed":
        raise RagError("Material must be processed before building a RAG index")
    return metadata


def _read_segments(metadata: dict[str, Any]) -> list[dict]:
    segments_path = metadata.get("segments_path")
    if not segments_path:
        raise RagError("Processed material has no segments path")

    path = Path(segments_path)
    if not path.exists():
        raise RagError("Segments file was not found")

    with path.open("r", encoding="utf-8") as file:
        segments = json.load(file)
    if not isinstance(segments, list) or not segments:
        raise RagError("Segments file is empty")
    return segments


def _index_paths(material_id: str) -> dict[str, Path]:
    index_path = Path(settings.VECTOR_DB_DIR) / material_id
    return {
        "index_path": index_path,
        "chunks_path": index_path / "chunks.json",
        "vectorizer_path": index_path / "tfidf_vectorizer.joblib",
        "matrix_path": index_path / "tfidf_matrix.joblib",
    }


def _index_files_exist(material_id: str) -> bool:
    paths = _index_paths(material_id)
    return (
        paths["chunks_path"].exists()
        and paths["vectorizer_path"].exists()
        and paths["matrix_path"].exists()
    )


def build_rag_index(material_id: str) -> dict:
    metadata = _load_processed_material(material_id)
    segments = _read_segments(metadata)
    chunks = chunk_segments(segments)
    if not chunks:
        raise RagError("No chunks could be created from this material")

    ensure_directories()
    paths = _index_paths(material_id)
    paths["index_path"].mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    with paths["chunks_path"].open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)
    joblib.dump(vectorizer, paths["vectorizer_path"])
    joblib.dump(matrix, paths["matrix_path"])

    metadata["rag"] = {
        "index_path": str(paths["index_path"]),
        "chunks_path": str(paths["chunks_path"]),
        "chunks_count": len(chunks),
        "retriever": "tfidf",
    }
    _save_material_metadata(metadata)

    return {
        "material_id": material_id,
        "status": "created",
        "retriever": "tfidf",
        "chunks_count": len(chunks),
    }


def get_rag_status(material_id: str) -> dict:
    metadata = _load_material(material_id)
    rag = metadata.get("rag") or {}
    rag_ready = bool(rag) and _index_files_exist(material_id)
    return {
        "material_id": material_id,
        "rag_ready": rag_ready,
        "retriever": rag.get("retriever") if rag_ready else None,
        "chunks_count": int(rag.get("chunks_count") or 0) if rag_ready else 0,
    }


def _load_index(material_id: str) -> tuple[list[dict], TfidfVectorizer, Any]:
    paths = _index_paths(material_id)
    if not _index_files_exist(material_id):
        build_rag_index(material_id)

    with paths["chunks_path"].open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    vectorizer = joblib.load(paths["vectorizer_path"])
    matrix = joblib.load(paths["matrix_path"])
    return chunks, vectorizer, matrix


def _select_sources(chunks: list[dict], scores: list[float], top_k: int) -> list[dict]:
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    selected = []
    for index in ranked_indices[: max(1, top_k)]:
        chunk = chunks[index]
        selected.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_start": chunk["source_start"],
                "source_end": chunk["source_end"],
                "score": round(float(scores[index]), 4),
            }
        )
    return selected


def _context_from_sources(chunks: list[dict], sources: list[dict]) -> str:
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    blocks = []
    for source in sources:
        chunk = chunks_by_id[source["chunk_id"]]
        blocks.append(
            f"{chunk['chunk_id']} ({chunk['source_start']} -> {chunk['source_end']}):\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _fallback_answer(chunks: list[dict], sources: list[dict]) -> str:
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    excerpts = []
    for source in sources:
        chunk = chunks_by_id[source["chunk_id"]]
        excerpt = " ".join(chunk["text"].split())
        excerpts.append(
            f"- {excerpt[:500]} ({chunk['source_start']} -> {chunk['source_end']})"
        )

    source_lines = [
        f"* {source['source_start']} -> {source['source_end']}" for source in sources
    ]
    return (
        "Ниже приведены наиболее релевантные фрагменты материала:\n\n"
        + "\n".join(excerpts)
        + "\n\nИсточники:\n"
        + "\n".join(source_lines)
    )


def ask_material_question(material_id: str, question: str, top_k: int = 4) -> dict:
    if not question.strip():
        raise RagError("Question must not be empty")

    _load_processed_material(material_id)
    chunks, vectorizer, matrix = _load_index(material_id)
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).flatten().tolist()
    sources = _select_sources(chunks, scores, top_k)

    if not (settings.OPENROUTER_API_KEY or "").strip():
        answer = _fallback_answer(chunks, sources)
    else:
        context = _context_from_sources(chunks, sources)
        prompt = f"""
Ты отвечаешь на вопросы только по предоставленному материалу.
Не используй внешние знания.
Если материал не содержит ответа, скажи:
"В загруженном материале я не нашёл ответа на этот вопрос."
Ответь на русском языке.
В конце добавь раздел:

Источники:
* source_start -> source_end

Вопрос:
{question}

Контекст:
\"\"\"
{context}
\"\"\"
""".strip()
        answer = LLMClient().generate(prompt)

    return {
        "material_id": material_id,
        "question": question,
        "answer": answer,
        "sources": sources,
    }
