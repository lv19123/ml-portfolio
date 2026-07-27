from pydantic import BaseModel


class Segment(BaseModel):
    text: str
    source: str
    start: str | None = None
    end: str | None = None


class Topic(BaseModel):
    title: str
    summary: str
    source_start: str
    source_end: str


class KeyTerm(BaseModel):
    term: str
    definition: str
    source: str


class RagSource(BaseModel):
    chunk_id: str
    source_start: str
    source_end: str
    score: float


class MaterialMetadata(BaseModel):
    material_id: str
    original_filename: str
    stored_filename: str | None = None
    file_extension: str
    source_type: str
    status: str
    source_url: str | None = None
    processed_dir: str | None = None
    segments_path: str | None = None
    extracted_text_path: str | None = None
    segments_count: int | None = None
    characters_count: int | None = None
    error_message: str | None = None
    reports: dict[str, dict[str, str]] | None = None
    topics_path: str | None = None
    topics_count: int | None = None
    terms_path: str | None = None
    terms_count: int | None = None
    rag: dict | None = None
    has_timestamps: bool | None = None
    timestamp_format: str | None = None
    transcription_engine: str | None = None


class UploadResponse(MaterialMetadata):
    pass


class YouTubeMaterialRequest(BaseModel):
    url: str


class YouTubeMaterialResponse(MaterialMetadata):
    pass


class ProcessResponse(MaterialMetadata):
    pass


class TextResponse(BaseModel):
    material_id: str
    text: str


class ReportResponse(BaseModel):
    material_id: str
    report_type: str
    status: str
    report_path: str


class ReportContentResponse(BaseModel):
    material_id: str
    report_type: str
    content: str


class ExportResponse(BaseModel):
    material_id: str
    report_type: str
    format: str
    status: str
    file_path: str


class TopicsResponse(BaseModel):
    material_id: str
    status: str
    topics_count: int
    topics: list[Topic]


class TermsResponse(BaseModel):
    material_id: str
    status: str
    terms_count: int
    terms: list[KeyTerm]


class RagBuildResponse(BaseModel):
    material_id: str
    status: str
    retriever: str
    chunks_count: int


class RagStatusResponse(BaseModel):
    material_id: str
    rag_ready: bool
    retriever: str | None = None
    chunks_count: int = 0


class AskQuestionRequest(BaseModel):
    question: str
    top_k: int = 4


class AskQuestionResponse(BaseModel):
    material_id: str
    question: str
    answer: str
    sources: list[RagSource]
