"""知识库切片与检索的请求/响应模型。"""

from typing import Literal

from pydantic import BaseModel, Field

from app_es.core.config import settings


class ChunkOut(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    vector: list[float] | None = None  # 仅 include_vectors=true 时返回


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    title: str | None = None
    doc_id: str | None = None  # 不传则自动生成 uuid4
    chunk_size: int | None = Field(default=None, ge=64, le=4096)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1024)


class IngestResponse(BaseModel):
    doc_id: str
    title: str | None
    chunk_count: int
    chunks: list[ChunkOut]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=settings.knowledge_top_k, ge=1, le=50)
    doc_ids: list[str] | None = None  # 限定召回范围（ES 查询层过滤）
    include_text: bool = True
    # vector = ES kNN 向量召回；keyword = ES BM25 关键词召回；
    # hybrid = 向量 + 关键词两路融合
    mode: Literal["vector", "keyword", "hybrid"] = "vector"
    # 混合检索中向量路的权重，关键词路取 1 - alpha
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    # 是否对召回候选做 cross-encoder 精排
    rerank: bool = False


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    score: float
    text: str
    title: str | None


class DocumentAggregate(BaseModel):
    doc_id: str
    title: str | None
    best_score: float
    hit_count: int
    chunk_count: int
    full_text: str  # 由源文档还原的整篇正文
    matched_chunks: list[SearchHit]


class SearchResponse(BaseModel):
    query: str
    top_k: int
    mode: str  # 实际使用的召回模式
    alpha: float  # 实际使用的向量路权重
    rerank: bool  # 是否启用了精排
    hits: list[SearchHit]  # 片段级召回
    documents: list[DocumentAggregate]  # 按文档聚合


class DocumentSummary(BaseModel):
    doc_id: str
    title: str | None
    chunk_count: int
    char_count: int


class DocumentDetail(BaseModel):
    doc_id: str
    title: str | None
    chunk_count: int
    full_text: str
    chunks: list[ChunkOut]


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool


class ESHealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    url: str
    cluster_name: str | None = None
    version: str | None = None
    documents_index: bool = False
    chunks_index: bool = False
