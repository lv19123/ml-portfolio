from openai import OpenAI

from backend.app.config import settings


class LLMClient:
    def __init__(self) -> None:
        self.api_key = (settings.OPENROUTER_API_KEY or "").strip()
        self.base_url = settings.OPENROUTER_BASE_URL
        self.model = settings.LLM_MODEL

    def generate(self, prompt: str) -> str:
        if self.api_key:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned an empty response")
            return content.strip()

        # Development/testing fallback only. This avoids external API calls when no
        # OpenRouter key is configured, so tests and local demos stay deterministic.
        if "Полный очищенный конспект" in prompt:
            return (
                "# Полный очищенный конспект\n\n"
                "## Локальный режим\n\n"
                "API ключ не настроен, поэтому создан тестовый конспект. "
                "Подключите OpenRouter, чтобы получить LLM-версию отчёта."
            )

        return (
            "# Краткий отчёт\n\n"
            "## О чём материал\n\n"
            "API ключ не настроен, поэтому создан тестовый краткий отчёт.\n\n"
            "## Основные темы\n\n"
            "- Локальная проверка пайплайна\n\n"
            "## Краткое содержание\n\n"
            "Материал был обработан без внешнего LLM-запроса.\n\n"
            "## Что важно запомнить\n\n"
            "Для реальной генерации укажите OPENROUTER_API_KEY."
        )
