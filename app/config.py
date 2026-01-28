from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "hybridqa-agents"
    data_dir: str = "data"
    sample_dir: str = "data/sample"
    index_dir: str = "data/index"

    llm_provider: str = "hf_local"
    hf_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_max_new_tokens: int = 512
    hf_temperature: float = 0.2
    hf_top_p: float = 0.9
    hf_trust_remote_code: bool = True
    hf_offline: bool = False

    openai_api_key: Optional[str] = None
    openai_model_planner: str = "gpt-4o"
    openai_model_table: str = "gpt-4o"
    openai_model_analysis: str = "gpt-4o"
    openai_json_mode: bool = True
    embeddings_model: str = "text-embedding-3-small"
    embeddings_provider: str = "hf"
    hf_embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    use_fake_embeddings: bool = False
    max_rag_results: int = 5
    rag_max_hops: int = 2
    rag_enable_multihop: bool = True
    request_timeout_s: int = 60
    rag_chunk_size: int = 400
    rag_chunk_overlap: int = 50
    allow_table_path: bool = False
    table_path_root: Optional[str] = None
    table_id_pattern: str = r"^[A-Za-z0-9_\-]+$"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HYBRIDQA_",
        extra="ignore",
    )


settings = Settings()
