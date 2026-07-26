import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from backend.app.config import ensure_directories, settings
from backend.app.schemas.materials import (
    AskQuestionRequest,
    AskQuestionResponse,
    ExportResponse,
    KeyTerm,
    MaterialMetadata,
    ProcessResponse,
    RagBuildResponse,
    RagStatusResponse,
    TermsResponse,
    ReportContentResponse,
    ReportResponse,
    Segment,
    TextResponse,
    Topic,
    TopicsResponse,
    UploadResponse,
    YouTubeMaterialRequest,
    YouTubeMaterialResponse,
)
from backend.app.services.file_utils import (
    detect_source_type,
    get_file_extension,
    validate_extension,
)
from backend.app.services.exporter import (
    ExportError,
    export_report_to_docx,
    export_report_to_pdf,
)
from backend.app.services.processing import (
    MaterialNotFoundError,
    ProcessingError,
    UnsupportedProcessingError,
    process_material,
)
from backend.app.services.rag import (
    RagError,
    ask_material_question,
    build_rag_index,
    get_rag_status,
)
from backend.app.services.report_generator import (
    ReportGenerationError,
    ReportNotFoundError,
    generate_full_clean_notes,
    generate_short_report,
    get_report_path,
    read_report,
)
from backend.app.services.structure_extractor import (
    StructuredOutputNotFoundError,
    StructureExtractionError,
    generate_key_terms,
    generate_topics,
    get_key_terms,
    get_topics,
)
from backend.app.services.storage import add_material, get_material
from backend.app.services.youtube_utils import extract_youtube_video_id

router = APIRouter(prefix="/materials", tags=["materials"])


@router.post("/upload", response_model=UploadResponse)
async def upload_material(file: UploadFile = File(...)) -> UploadResponse:
    extension = get_file_extension(file.filename or "")
    if not validate_extension(extension):
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    ensure_directories()
    material_id = str(uuid4())
    stored_filename = f"{material_id}{extension}"
    destination = Path(settings.UPLOADS_DIR) / stored_filename

    content = await file.read()
    destination.write_bytes(content)

    metadata = MaterialMetadata(
        material_id=material_id,
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        file_extension=extension,
        source_type=detect_source_type(extension),
        status="uploaded",
    )
    add_material(metadata.model_dump(exclude_none=True))

    # TODO: Later stages can add richer pipeline orchestration.
    return UploadResponse(**metadata.model_dump())


@router.post("/youtube", response_model=YouTubeMaterialResponse)
def add_youtube_material(request: YouTubeMaterialRequest) -> YouTubeMaterialResponse:
    try:
        video_id = extract_youtube_video_id(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ensure_directories()
    metadata = MaterialMetadata(
        material_id=str(uuid4()),
        original_filename=video_id or "youtube_video",
        stored_filename=None,
        file_extension=".youtube",
        source_type="youtube",
        source_url=request.url,
        status="uploaded",
    )
    add_material(metadata.model_dump(exclude_none=True))
    return YouTubeMaterialResponse(**metadata.model_dump())


@router.get("/{material_id}/status", response_model=MaterialMetadata)
def material_status(material_id: str) -> MaterialMetadata:
    metadata = get_material(material_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Material not found")
    return MaterialMetadata(**metadata)


@router.post("/{material_id}/process", response_model=ProcessResponse)
def process_material_endpoint(material_id: str) -> ProcessResponse:
    try:
        metadata = process_material(material_id)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnsupportedProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected processing error") from exc

    return ProcessResponse(**metadata)


@router.get("/{material_id}/segments", response_model=list[Segment])
def material_segments(material_id: str) -> list[Segment]:
    metadata = get_material(material_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Material not found")
    if metadata.get("status") != "processed" or not metadata.get("segments_path"):
        raise HTTPException(status_code=400, detail="Material is not processed yet")

    segments_path = Path(metadata["segments_path"])
    if not segments_path.exists():
        raise HTTPException(status_code=400, detail="Material is not processed yet")

    with segments_path.open("r", encoding="utf-8") as file:
        segments = json.load(file)
    return [Segment(**segment) for segment in segments]


@router.get("/{material_id}/text", response_model=TextResponse)
def material_text(material_id: str) -> TextResponse:
    metadata = get_material(material_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Material not found")
    if metadata.get("status") != "processed" or not metadata.get("extracted_text_path"):
        raise HTTPException(status_code=400, detail="Material is not processed yet")

    extracted_text_path = Path(metadata["extracted_text_path"])
    if not extracted_text_path.exists():
        raise HTTPException(status_code=400, detail="Material is not processed yet")

    return TextResponse(
        material_id=material_id,
        text=extracted_text_path.read_text(encoding="utf-8"),
    )


@router.post("/{material_id}/topics/generate", response_model=TopicsResponse)
def generate_material_topics(material_id: str) -> TopicsResponse:
    try:
        result = generate_topics(material_id)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StructureExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected topic extraction error") from exc

    return TopicsResponse(**result)


@router.get("/{material_id}/topics", response_model=list[Topic])
def material_topics(material_id: str) -> list[Topic]:
    try:
        return [Topic(**topic) for topic in get_topics(material_id)]
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StructuredOutputNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StructureExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{material_id}/terms/generate", response_model=TermsResponse)
def generate_material_terms(material_id: str) -> TermsResponse:
    try:
        result = generate_key_terms(material_id)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StructureExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected key term extraction error") from exc

    return TermsResponse(**result)


@router.get("/{material_id}/terms", response_model=list[KeyTerm])
def material_terms(material_id: str) -> list[KeyTerm]:
    try:
        return [KeyTerm(**term) for term in get_key_terms(material_id)]
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StructuredOutputNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StructureExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{material_id}/rag/build", response_model=RagBuildResponse)
def build_material_rag_index(material_id: str) -> RagBuildResponse:
    try:
        result = build_rag_index(material_id)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected RAG index error") from exc

    return RagBuildResponse(**result)


@router.get("/{material_id}/rag/status", response_model=RagStatusResponse)
def material_rag_status(material_id: str) -> RagStatusResponse:
    try:
        result = get_rag_status(material_id)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return RagStatusResponse(**result)


@router.post("/{material_id}/ask", response_model=AskQuestionResponse)
def ask_material(
    material_id: str,
    request: AskQuestionRequest,
) -> AskQuestionResponse:
    try:
        result = ask_material_question(material_id, request.question, request.top_k)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected question answering error") from exc

    return AskQuestionResponse(**result)


@router.post("/{material_id}/reports/short", response_model=ReportResponse)
def create_short_report(material_id: str) -> ReportResponse:
    try:
        report = generate_short_report(material_id)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReportGenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected report generation error") from exc

    return ReportResponse(**report)


@router.post("/{material_id}/reports/full-clean", response_model=ReportResponse)
def create_full_clean_report(material_id: str) -> ReportResponse:
    try:
        report = generate_full_clean_notes(material_id)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReportGenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected report generation error") from exc

    return ReportResponse(**report)


@router.get("/{material_id}/reports/{report_type}", response_model=ReportContentResponse)
def material_report(material_id: str, report_type: str) -> ReportContentResponse:
    try:
        report = read_report(material_id, report_type)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReportNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReportGenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ReportContentResponse(**report)


@router.get("/{material_id}/download/md")
def download_markdown_report(
    material_id: str,
    report_type: str = Query(...),
) -> FileResponse:
    try:
        report_path = get_report_path(material_id, report_type)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReportNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReportGenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(
        path=report_path,
        filename=report_path.name,
        media_type="text/markdown",
    )


@router.post("/{material_id}/exports/pdf", response_model=ExportResponse)
def create_pdf_export(
    material_id: str,
    report_type: str = Query(...),
) -> ExportResponse:
    try:
        export_path = export_report_to_pdf(material_id, report_type)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected PDF export error") from exc

    return ExportResponse(
        material_id=material_id,
        report_type=report_type,
        format="pdf",
        status="created",
        file_path=str(export_path),
    )


@router.post("/{material_id}/exports/docx", response_model=ExportResponse)
def create_docx_export(
    material_id: str,
    report_type: str = Query(...),
) -> ExportResponse:
    try:
        export_path = export_report_to_docx(material_id, report_type)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected DOCX export error") from exc

    return ExportResponse(
        material_id=material_id,
        report_type=report_type,
        format="docx",
        status="created",
        file_path=str(export_path),
    )


@router.get("/{material_id}/download/pdf")
def download_pdf_report(
    material_id: str,
    report_type: str = Query(...),
) -> FileResponse:
    try:
        export_path = export_report_to_pdf(material_id, report_type)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected PDF export error") from exc

    return FileResponse(
        path=export_path,
        filename=export_path.name,
        media_type="application/pdf",
    )


@router.get("/{material_id}/download/docx")
def download_docx_report(
    material_id: str,
    report_type: str = Query(...),
) -> FileResponse:
    try:
        export_path = export_report_to_docx(material_id, report_type)
    except MaterialNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected DOCX export error") from exc

    return FileResponse(
        path=export_path,
        filename=export_path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
