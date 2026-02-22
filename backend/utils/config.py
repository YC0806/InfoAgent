from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    firecrawl_api_key: str = ""
    brave_api_key: str = ""
    crawl4ai_base_url: str = ""
    mineru_api_key: str = ""
    mineru_base_url: str = ""
    logfire_enabled: bool = False
    logfire_token: str = ""

    class Config:
        env_file = Path(__file__).resolve().parents[1] / ".env"


settings = Settings()
