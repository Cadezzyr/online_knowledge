from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Online Knowledge API"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # 知识库切片与检索
    knowledge_store_dir: str = "data/knowledge"
    knowledge_chunk_size: int = 512  # 以字符计（见自定义 tokenizer）
    knowledge_chunk_overlap: int = 64
    knowledge_top_k: int = 5

    # 向量化
    embedding_model: str = "bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8

    # 精排（rerank）
    rerank_model: str = r"D:/Pycharm Projects/联网知识库情况/online_knowledge/data/models/bge-reranker-v2-m3"  # 本地目录或 HuggingFace 模型名
    rerank_top_n: int = 50  # 单次精排保留的候选数，最终仍按 top_k 截断


settings = Settings()
