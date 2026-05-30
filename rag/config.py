# rag/config.py — centralised settings with env validation
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter API key
    OPENROUTER_API_KEY: str = ""

    # Chunking parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_PAGES: int = 20

    # Paths
    ARTIFACTS_DIR: str = "./Artifacts"
    VECTOR_STORE_PATH: str = "./vector_store"

    # LLM model on OpenRouter (first available is used; fallbacks tried on 429)
    LLM_MODEL: str = "openai/gpt-oss-120b:free"
    LLM_FALLBACK_MODELS: list[str] = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3-coder:free",
        "openrouter/owl-alpha",
    ]
    MAX_TOKENS: int = 200
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"


# Single shared instance — import this everywhere
settings = Settings()
