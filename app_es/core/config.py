from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Online Knowledge API (Elasticsearch)"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # 知识库切片与检索
    knowledge_chunk_size: int = 512  # 以字符计（见自定义 tokenizer）
    knowledge_chunk_overlap: int = 64
    knowledge_top_k: int = 5
    knowledge_list_limit: int = 500  # 文档列表/聚合单次最大条数

    # Elasticsearch 连接
    es_url: str = "http://localhost:9200"
    es_username: str = "elastic"
    es_password: str = "123456"
    es_api_key: str | None = None
    es_verify_certs: bool = True
    es_ca_certs: str | None = None
    es_request_timeout: float = 10.0

    # Elasticsearch 索引与映射
    es_index_documents: str = "knowledge_documents"
    es_index_chunks: str = "knowledge_chunks"
    es_analyzer: str = "standard"  # 中文可改 ik_max_word（需先装 IK 插件）
    es_number_of_shards: int = 1
    es_number_of_replicas: int = 0
    es_embedding_dim: int = 512  # 必须与 embedding 模型维度一致

    # 向量化
    embedding_model: str = "bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8

    # 精排（rerank）
    rerank_model: str = r"D:/Pycharm Projects/联网知识库情况/online_knowledge/data/models/bge-reranker-v2-m3"  # 本地目录或 HuggingFace 模型名
    rerank_top_n: int = 50  # 单次精排保留的候选数，最终仍按 top_k 截断


settings = Settings()
