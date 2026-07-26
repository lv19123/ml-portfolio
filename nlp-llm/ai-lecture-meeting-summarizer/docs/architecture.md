# Architecture

## Project Goal

AI Lecture / Meeting Summarizer is a pipeline application for turning learning and meeting materials into usable study artifacts. It accepts documents, audio/video files, and YouTube URLs with available transcripts, then produces extracted text, structured reports, topics, key terms, searchable context, and downloadable exports.

## Not An Agent

This project is intentionally not an autonomous agent. It does not plan multi-step actions on its own, browse for missing information, or fine-tune models. Each user action triggers a deterministic pipeline stage:

```text
upload file / YouTube URL
-> detect source type
-> extract text or transcript
-> save segments
-> generate reports
-> extract topics and terms
-> build RAG index
-> answer questions
-> export Markdown / PDF / DOCX
```

## Main Modules

- `backend/app/api/`: FastAPI routes for upload, processing, reports, exports, topic extraction, RAG, and YouTube materials.
- `backend/app/services/`: Pipeline services for storage, extraction, transcription, report generation, RAG, exports, and utility logic.
- `frontend/app.py`: Streamlit UI for uploading materials, processing them, generating outputs, and asking questions.
- `data/uploads/`: Uploaded source files.
- `data/processed/`: Extracted text, segments, topics, and key terms.
- `data/reports/`: Markdown, PDF, and DOCX reports.
- `data/vector_db/`: Local TF-IDF chunks, vectorizer, and matrix files for RAG.

## AI Components

- `faster-whisper`: Speech-to-text for uploaded audio/video materials.
- OpenRouter-compatible LLM API: Used for short reports, full cleaned notes, topics, key terms, and RAG answers when an API key is configured.
- Deterministic fallback mode: Keeps local development and tests working without API keys, model downloads, or internet access.
- Local TF-IDF RAG: Uses scikit-learn for an MVP retriever with no external embedding model.

## Future Upgrades

- Replace TF-IDF with embeddings plus ChromaDB or FAISS.
- Add background jobs for long media processing.
- Add map-reduce summarization for long documents and transcripts.
- Add authentication and multi-user storage.
- Improve frontend layout for larger production workflows.
