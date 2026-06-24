"""RAG index loading and retrieval for local materials."""

import logging

from langchain_core.documents import Document

from app.config import CHROMA_DIR, MATERIALS_DIR, RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, RAG_TOP_K


logger = logging.getLogger(__name__)
_rag_retriever = None


def _load_documents() -> list[Document]:
    """Загружает все .txt, .md, .pdf из MATERIALS_DIR и подпапок."""
    if not MATERIALS_DIR.is_dir():
        return []
    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    documents = []
    for ext in ("*.txt", "*.md"):
        for path in MATERIALS_DIR.rglob(ext):
            try:
                docs = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
                documents.extend(docs)
            except Exception as e:
                logger.warning("Не удалось загрузить текстовый материал %s: %s", path, e)
    for path in MATERIALS_DIR.rglob("*.pdf"):
        try:
            documents.extend(PyPDFLoader(str(path)).load())
        except Exception as e:
            logger.warning("Не удалось загрузить PDF-материал %s: %s", path, e)
    return documents


def _get_rag_retriever():
    """Лениво загружает индекс с диска или строит из папки materials."""
    global _rag_retriever
    if _rag_retriever is not None:
        return _rag_retriever if _rag_retriever else None
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError as e:
        logger.warning("RAG недоступен: не удалось импортировать зависимости: %s", e)
        _rag_retriever = False
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    if (CHROMA_DIR / "chroma.sqlite3").exists():
        try:
            vectorstore = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
            )
            _rag_retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K})
            return _rag_retriever
        except Exception as e:
            logger.warning("Не удалось загрузить Chroma-индекс из %s: %s. Пробую пересобрать индекс.", CHROMA_DIR, e)

    documents = _load_documents()
    if not documents:
        logger.warning("RAG недоступен: материалы не найдены или не удалось загрузить ни один файл из %s.", MATERIALS_DIR)
        _rag_retriever = False
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        length_function=len,
    )
    splits = splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    _rag_retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K})
    logger.info("RAG: индекс по материалам собран, можно спрашивать по конспектам.")
    return _rag_retriever


def get_rag_context(question: str) -> str:
    """Возвращает релевантные куски из материалов для вопроса, или пустую строку."""
    retriever = _get_rag_retriever()
    if retriever is None:
        return ""
    try:
        docs = retriever.invoke(question)
        if not docs:
            return ""
        return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)
    except Exception as e:
        logger.warning("Не удалось получить RAG-контекст для вопроса: %s", e)
        return ""
