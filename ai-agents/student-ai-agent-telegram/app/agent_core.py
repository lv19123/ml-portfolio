"""LangChain tool-calling agent and ask_ai entrypoint."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.config import OPENROUTER_API_KEY
from app.memory import (
    CONVERSATIONS,
    RECENT_MESSAGES_KEEP,
    SUMMARIES,
    SUMMARIZE_WHEN_OVER,
    _summarize_conversation_chunk,
)
from app.rag import get_rag_context
from app.tools.basic import calculator, get_current_time
from app.tools.jobs import get_ml_ai_jobs
from app.tools.profi import get_profi_orders
from app.tools.schedule import get_schedule_tool, get_tomorrow_schedule_tool


SYSTEM_PROMPT = (
    "Ты — полезный ассистент студента МИЭТ с доступом к инструментам. "
    "Источники ответа: если в контексте есть «Релевантные фрагменты из материалов» и они относятся к вопросу — опирайся на них. На остальные вопросы (география, наука, факты и т.д.) отвечай из своих знаний, не отказывайся. "
    "Инструменты вызывай только когда нужны: расписание (get_schedule_tool), вакансии (get_ml_ai_jobs), заказы (get_profi_orders), вычисления (calculator), время (get_current_time). "
    "Расписание: при запросе про расписание обязательно вызови get_schedule_tool(day=\"сегодня\") или get_schedule_tool(day=\"завтра\") и включи в ответ ПОЛНЫЙ текст инструмента (список пар или «пар нет»). "
    "Вакансии: при запросе «вакансии» / «покажи вакансии» сразу вызывай get_ml_ai_jobs(count=\"10\", for_beginners=\"да\") и выводи результат целиком. Заказы: при запросе про заказы — get_profi_orders(count=\"10\"). "
    "Отвечай по‑русски, кратко и по делу."
)

TOOLS_LIST = [calculator, get_current_time, get_schedule_tool, get_tomorrow_schedule_tool, get_ml_ai_jobs, get_profi_orders]
TOOLS_BY_NAME = {t.name: t for t in TOOLS_LIST}


def _run_agent_loop(input_text: str) -> str:
    """Цикл агента: LLM с bind_tools -> выполнение инструментов -> повторный вызов LLM."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("AI‑агент не настроен: нет OPENROUTER_API_KEY в .env.")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model="google/gemini-2.0-flash-001",
        temperature=0,
    ).bind_tools(TOOLS_LIST)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=input_text),
    ]
    max_steps = 12
    for _ in range(max_steps):
        response = llm.invoke(messages)
        if not getattr(response, "tool_calls", None):
            return (response.content or "").strip()
        messages.append(response)
        for tc in response.tool_calls:
            name = tc.get("name") or (getattr(tc, "name", None))
            args = tc.get("args") or getattr(tc, "args", {}) or {}
            tid = tc.get("id") or getattr(tc, "id", "")
            selected_tool = TOOLS_BY_NAME.get(name) if name else None
            if not selected_tool:
                out = f"Неизвестный инструмент: {name}"
            else:
                try:
                    if isinstance(args, dict):
                        out = selected_tool.invoke(args)
                    else:
                        out = str(selected_tool.invoke(args))
                except Exception as e:
                    out = f"Ошибка: {e}"
            messages.append(ToolMessage(content=out, tool_call_id=tid))
    return "Превышено число шагов. Попробуй переформулировать вопрос."


def build_agent():
    """Возвращает объект с методом invoke({"input": ...}) -> {"output": ...}."""
    class Executor:
        def invoke(self, inp: dict):
            return {"output": _run_agent_loop(inp.get("input", ""))}

    return Executor()


def ask_ai(chat_id: int, question: str) -> str:
    """Вопрос к агенту: RAG + резюме диалога + последние реплики."""
    conv = CONVERSATIONS.get(chat_id, [])
    summary = SUMMARIES.get(chat_id, "")

    rag_context = get_rag_context(question)

    parts = []
    if rag_context:
        parts.append(
            "Релевантные фрагменты из материалов пользователя (конспекты, лекции). "
            "Опирайся на них в первую очередь при ответе.\n\n" + rag_context
        )
    if summary:
        parts.append(f"Краткое резюме предыдущего диалога:\n{summary}")
    recent = conv[-RECENT_MESSAGES_KEEP:] if len(conv) > RECENT_MESSAGES_KEEP else conv
    if recent:
        parts.append("Последние реплики:\n" + "\n".join(recent))
    if parts:
        full_input = "\n\n".join(parts) + f"\n\nНовый вопрос:\n{question}"
    else:
        full_input = question

    try:
        agent_executor = build_agent()
        result = agent_executor.invoke({"input": full_input})
        answer = result.get("output", "").strip()
    except Exception as e:
        return f"Не удалось получить ответ от AI‑агента: {e}"

    conv = CONVERSATIONS.get(chat_id, [])
    conv.append(f"Ты: {question}")
    conv.append(f"Ассистент: {answer}")

    if len(conv) > SUMMARIZE_WHEN_OVER:
        to_summarize = "\n".join(conv[:-RECENT_MESSAGES_KEEP])
        new_summary = _summarize_conversation_chunk(to_summarize)
        if new_summary:
            SUMMARIES[chat_id] = new_summary
        CONVERSATIONS[chat_id] = conv[-RECENT_MESSAGES_KEEP:]
    else:
        CONVERSATIONS[chat_id] = conv

    return answer
