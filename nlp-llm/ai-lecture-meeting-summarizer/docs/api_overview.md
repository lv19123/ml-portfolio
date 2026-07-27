# API Overview

## Health

- `GET /health`

Use this to verify that the backend is running.

## Materials

- `POST /materials/upload`
- `POST /materials/youtube`
- `GET /materials/{material_id}/status`
- `POST /materials/{material_id}/process`
- `GET /materials/{material_id}/text`
- `GET /materials/{material_id}/segments`

Use these endpoints to create a material from a file or YouTube URL, process it, and inspect extracted text or timestamped/source-aware segments.

## Reports

- `POST /materials/{material_id}/reports/short`
- `POST /materials/{material_id}/reports/full-clean`
- `GET /materials/{material_id}/reports/{report_type}`

Use these after processing a material to generate a compact report or full cleaned notes in Markdown.

## Exports

- `GET /materials/{material_id}/download/md`
- `GET /materials/{material_id}/download/pdf`
- `GET /materials/{material_id}/download/docx`

Use these endpoints to download generated reports. PDF and DOCX exports are created from existing Markdown reports.

## Topics And Terms

- `POST /materials/{material_id}/topics/generate`
- `GET /materials/{material_id}/topics`
- `POST /materials/{material_id}/terms/generate`
- `GET /materials/{material_id}/terms`

Use these endpoints to extract ordered topics with source references and key terms with short definitions.

## RAG

- `POST /materials/{material_id}/rag/build`
- `GET /materials/{material_id}/rag/status`
- `POST /materials/{material_id}/ask`

Use these endpoints to build a local TF-IDF retrieval index and ask grounded questions about one processed material.
