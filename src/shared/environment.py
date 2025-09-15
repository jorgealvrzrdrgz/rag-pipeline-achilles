from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENAI_API_KEY: str

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "rag-pipeline"

    EMBEDDINGS_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDINGS_TOKENIZER: str = "Qwen/Qwen3-Embedding-0.6B"

    RERANKER_MODEL: str = "Alibaba-NLP/gte-multilingual-reranker-base"

    RERANK: bool = True

    PDFS_URLS: list[str] = [
        "https://arxiv.org/pdf/1706.03762",
        "https://arxiv.org/pdf/1810.04805",
        "https://arxiv.org/pdf/1906.08237",
        "https://arxiv.org/pdf/1909.11942",
        "https://arxiv.org/pdf/2302.13971"
    ]