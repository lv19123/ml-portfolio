def chunk_segments(
    segments: list[dict],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[dict]:
    chunks = []
    current_parts = []
    current_size = 0
    current_source_start = None
    current_source_end = None

    def add_chunk(text: str, source_start: str, source_end: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        chunks.append(
            {
                "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                "text": stripped,
                "source_start": source_start,
                "source_end": source_end,
            }
        )

    def flush_current() -> None:
        nonlocal current_parts, current_size, current_source_start, current_source_end
        if current_parts:
            add_chunk(
                "\n\n".join(current_parts),
                current_source_start or "text",
                current_source_end or current_source_start or "text",
            )
        current_parts = []
        current_size = 0
        current_source_start = None
        current_source_end = None

    step = max(1, chunk_size - chunk_overlap)
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        source_start = str(segment.get("start") or segment.get("source") or "text")
        source_end = str(segment.get("end") or segment.get("source") or source_start)
        if len(text) > chunk_size:
            flush_current()
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                add_chunk(text[start:end], source_start, source_end)
                if end == len(text):
                    break
                start += step
            continue

        separator_size = 2 if current_parts else 0
        if current_parts and current_size + separator_size + len(text) > chunk_size:
            flush_current()

        if not current_parts:
            current_source_start = source_start
        current_parts.append(text)
        current_size += separator_size + len(text)
        current_source_end = source_end

    flush_current()
    return chunks
