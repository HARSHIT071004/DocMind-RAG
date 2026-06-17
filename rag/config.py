from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = ""

    SMALL_CHUNK_SIZE: int = 300
    SMALL_CHUNK_OVERLAP: int = 50
    PARENT_CHUNK_SIZE: int = 1000
    PARENT_CHUNK_OVERLAP: int = 200
    MAX_PAGES: int = 20

    ARTIFACTS_DIR: str = "./Artifacts"
    VECTOR_STORE_PATH: str = "./vector_store"

    LLM_MODEL: str = "openai/gpt-oss-120b:free"
    LLM_FALLBACK_MODELS: list[str] = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3-coder:free",
        "openrouter/owl-alpha",
    ]
    MAX_TOKENS: int = 200

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    TOP_K_FAISS: int = 12
    TOP_K_BM25: int = 12
    TOP_K_HYBRID: int = 4
    RRF_K: int = 60
    RRF_DENSE_WEIGHT: float = 0.7
    RRF_SPARSE_WEIGHT: float = 0.3
    MMR_LAMBDA: float = 0.7
    MMR_NUM_DIVERSIFY: int = 8
    ENABLE_CROSS_ENCODER: bool = True
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"

    CACHE_MAXSIZE: int = 128
    CACHE_TTL_SECONDS: int = 300
    MEMORY_MAX_TURNS: int = 4
    MEMORY_MAX_TOKENS: int = 1000
    CONFIDENCE_THRESHOLD: float = 0.5
    CRAG_ENABLED: bool = True
    CRAG_RETRY_MODEL: str = "openai/gpt-oss-120b:free"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_TIMEOUT_SECONDS: int = 15

    class Config:
        env_file = ".env"


settings = Settings()
