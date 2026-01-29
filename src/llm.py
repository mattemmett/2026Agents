import os
from typing import Optional

from langchain_openai import ChatOpenAI


def get_llm(model: Optional[str] = None, temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ["LLM_API_KEY"]
    base_url = os.getenv("LLM_BASE_URL", "https://llm.ai.syntax-rnd.com").rstrip("/")

    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "gpt-4o"),
        base_url=base_url,
        # Your gateway checks Bearer format, so this must be the real sk- key.
        api_key=api_key,
        # Your gateway also expects x-api-key.
        default_headers={"x-api-key": api_key},
        temperature=temperature,
    )