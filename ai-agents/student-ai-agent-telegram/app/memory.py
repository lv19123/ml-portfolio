"""In-memory conversation history and summary memory."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import OPENROUTER_API_KEY


CONVERSATIONS: dict[int, list[str]] = {}
SUMMARIES: dict[int, str] = {}

RECENT_MESSAGES_KEEP = 6
SUMMARIZE_WHEN_OVER = 10


def _summarize_conversation_chunk(text: str) -> str:
    """Сжимает старую часть диалога в 2-4 предложения."""
    if not text.strip():
        return ""
    if not OPENROUTER_API_KEY:
        return ""

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model="google/gemini-2.0-flash-001",
        temperature=0,
        max_tokens=300,
    )
    messages = [
        SystemMessage(content="Ты сжимаешь фрагмент диалога в краткое резюме на русском: 2-4 предложения. Сохрани важное: имена, темы, решения, числа."),
        HumanMessage(content=text),
    ]
    try:
        response = llm.invoke(messages)
        return (response.content or "").strip()
    except Exception:
        return ""
