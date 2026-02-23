from __future__ import annotations

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import ModelSettings

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def get_model() -> OpenAIChatModel:
    """Create a PydanticAI model configured for the DeepSeek OpenAI-compatible endpoint."""
    logger.info("Initializing OpenAI model: %s, base_url: %s", settings.llm_model, settings.llm_base_url)
    model = OpenAIChatModel(
        settings.llm_model,
        provider=OpenAIProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key
        ),
        settings=ModelSettings(
            temperature=0.1,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )
    )
    return model