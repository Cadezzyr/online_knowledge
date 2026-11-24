"""基于 Elasticsearch 的分片知识库服务。

文档原文落 `knowledge_documents`，分片文本 + 向量落 `knowledge_chunks`；
向量、关键词、混合三种召回都由 ES 完成，混合模式用 llama_index 做加权融合。
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable
from uuid import uuid4

from elasticsearch.exceptions import ConnectionError as ESConnectionError
from elasticsearch.exceptions import TransportError
from elasticsearch.helpers import bulk
from llama_index.core import Document, Settings
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.utils import split_by_regex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app_es.core.config import settings
from app_es.core.es import (
    ElasticsearchUnavailableError,
    ensure_indices,
    get_es_client,
)
from app_es.schemas.knowledge import (
    ChunkOut,
    DocumentAggregate,
    DocumentDetail,
    DocumentSummary,
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app_es.services.retrievers import ESKeywordRetriever, ESVectorRetriever


class DocumentExistsError(Exception):
    """doc_id 已存在。"""


class DocumentNotFoundError(Exception):
    """doc_id 不存在。"""


class RerankerUnavailableError(Exception):
    """精排模型未就绪（模型目录缺失或加载失败）。"""


class EmbeddingDimensionError(Exception):
    """向量维度与 ES 映射不一致。"""


def _char_tokenizer(text: str) -> list[str]:
    """按字符计数，避免 SentenceSplitter 默认 tokenizer 联网下载编码表。"""
    return list(text)


_SENTENCE_REGEX = r"[^,.;。？！]+[,.;。？！]?|[,.;。？！]"


def _es_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """把 ES 连接/传输异常统一转成 ElasticsearchUnavailableError。"""

    @wraps(fn)
    def wrapper(self: "KnowledgeService", *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        except (ESConnectionError, TransportError) as exc:
            raise ElasticsearchUnavailableError(str(exc)) from exc

    return wrapper


class KnowledgeService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._client = get_es_client()
        self._embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            embed_batch_size=settings.embedding_batch_size,
        )
        # 精排模型按需加载，首次请求 rerank=true 时才占用内存
        self._reranker: SentenceTransformerRerank | None = None
        # 索引只需创建一次；失败时不置位，下次请求会重试
        self._ready = False
        # QueryFusionRetriever 构造时会读取 Settings.llm，
        # num_queries=1 时不会被调用，这里指向 MockLLM 避免依赖 LLM 供应商包。
        Settings.llm = MockLLM()

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        ensure_indices(self._client)
        self._ready = True

    def _get_reranker(self) -> SentenceTransformerRerank:
        """懒加载 cross-encoder 精排模型。"""
        if self._reranker is None:
            try:
                self._reranker = SentenceTransformerRerank(
                    model=settings.rerank_model,
                    top_n=settings.rerank_top_n,
                    device=settings.embedding_device,
                )
            except Exception as exc:
                raise RerankerUnavailableError(settings.rerank_model) from exc
        return self._reranker

    def _make_postprocessors(
        self, req: SearchRequest
    ) -> list[BaseNodePostprocessor]:
        if not req.rerank:
            return []
        return [self._get_reranker()]

    def _make_retriever(
        self, req: SearchRequest, candidates: int
    ) -> ESVectorRetriever | ESKeywordRetriever | QueryFusionRetriever:
        """按 mode 组装向量 / 关键词 / 混合召回。"""
        index_name = settings.es_index_chunks
        if req.mode == "vector":
            return ESVectorRetriever(
                self._client,
                index_name,
                self._embed_model,
                candidates,
                req.doc_ids,
            )

        keyword = ESKeywordRetriever(
            self._client, index_name, candidates, req.doc_ids
        )
        if req.mode == "keyword":
            return keyword

        vector = ESVectorRetriever(
            self._client,
            index_name,
            self._embed_model,
            candidates,
            req.doc_ids,
        )
        return QueryFusionRetriever(
            [vector, keyword],
            llm=MockLLM(),
            # 融合后保留候选池供后处理（如 rerank）精排，最终截断交给 search()
            similarity_top_k=candidates,
            num_queries=1,  # 不启用 LLM 查询改写
            # RRF 不读取 retriever_weights，要用 alpha 必须选相对分数融合
            mode=FUSION_MODES.RELATIVE_SCORE,
            use_async=False,
            retriever_weights=[req.alpha, 1.0 - req.alpha],
        )

    def _make_splitter(
        self, chunk_size: int | None, chunk_overlap: int | None
    ) -> SentenceSplitter:
        size = chunk_size or settings.knowledge_chunk_size
        overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.knowledge_chunk_overlap
        )
        # overlap 必须小于 chunk_size
        overlap = min(overlap, max(size - 1, 0))
        return SentenceSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            paragraph_separator="\n\n",
            tokenizer=_char_tokenizer,
            # 用正则切句，避免默认的 nltk punkt 在首次调用时联网下载
            chunking_tokenizer_fn=split_by_regex(_SENTENCE_REGEX),
        )

    @_es_guard
    def ingest(self, req: IngestRequest) -> IngestResponse:
        self._ensure_ready()
        doc_id = req.doc_id or str(uuid4())
        splitter = self._make_splitter(req.chunk_size, req.chunk_overlap)

        with self._lock:
            if self._client.exists(
                index=settings.es_index_documents, id=doc_id
            ):
                raise DocumentExistsError(doc_id)

            metadata = {"doc_id": doc_id}
            if req.title is not None:
                metadata["title"] = req.title
            document = Document(text=req.text, id_=doc_id, metadata=metadata)

            nodes = splitter.get_nodes_from_documents([document])
            for index, node in enumerate(nodes):
                node.metadata["chunk_index"] = index

            texts = [node.get_content() for node in nodes]
            embeddings = self._embed_model.get_text_embedding_batch(
                texts, show_progress=False
            )
            if embeddings and len(embeddings[0]) != settings.es_embedding_dim:
                raise EmbeddingDimensionError(
                    f"embedding 维度 {len(embeddings[0])} 与 "
                    f"ES_EMBEDDING_DIM={settings.es_embedding_dim} 不一致"
                )

            now = datetime.now(timezone.utc).isoformat()
            actions: list[dict[str, Any]] = []
            for node, vector in zip(nodes, embeddings):
                chunk_index = int(node.metadata["chunk_index"])
                actions.append(
                    {
                        "_op_type": "index",
                        "_index": settings.es_index_chunks,
                        "_id": node.node_id,
                        "_source": {
                            "chunk_id": node.node_id,
                            "doc_id": doc_id,
                            "ref_doc_id": doc_id,
                            "chunk_index": chunk_index,
                            "title": req.title,
                            "content": node.get_content(),
                            "embedding": vector,
                            "created_at": now,
                        },
                    }
                )
            actions.append(
                {
                    "_op_type": "index",
                    "_index": settings.es_index_documents,
                    "_id": doc_id,
                    "_source": {
                        "doc_id": doc_id,
                        "title": req.title,
                        "full_text": req.text,
                        "char_count": len(req.text),
                        "chunk_count": len(nodes),
                        "created_at": now,
                    },
                }
            )
            # refresh=wait_for 保证接口返回后可立即检索到
            bulk(self._client, actions, refresh="wait_for")

        chunks = [
            ChunkOut(
                chunk_id=node.node_id,
                chunk_index=int(node.metadata["chunk_index"]),
                text=node.get_content(),
            )
            for node in nodes
        ]
        return IngestResponse(
            doc_id=doc_id,
            title=req.title,
            chunk_count=len(chunks),
            chunks=chunks,
        )

    @_es_guard
    def search(self, req: SearchRequest) -> SearchResponse:
        self._ensure_ready()
        # 候选池要给 rerank 留素材，先放宽到 top_k 的 4 倍（至少 20）
        candidates = max(req.top_k * 4, 20)
        retriever = self._make_retriever(req, candidates)

        query_bundle = QueryBundle(req.query)
        with self._lock:
            nodes = retriever.retrieve(query_bundle)
            for processor in self._make_postprocessors(req):
                nodes = processor.postprocess_nodes(
                    nodes, query_bundle=query_bundle
                )
            results: list[NodeWithScore] = nodes[: req.top_k]

            hits: list[SearchHit] = []
            for result in results:
                node = result.node
                doc_id = node.ref_doc_id or node.metadata.get("doc_id")
                if doc_id is None:
                    continue
                hits.append(
                    SearchHit(
                        chunk_id=node.node_id,
                        doc_id=str(doc_id),
                        chunk_index=int(node.metadata.get("chunk_index", 0)),
                        score=float(result.score or 0.0),
                        text=node.get_content() if req.include_text else "",
                        title=node.metadata.get("title"),
                    )
                )

            documents = self._aggregate(hits)

        return SearchResponse(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            alpha=req.alpha,
            rerank=req.rerank,
            hits=hits,
            documents=documents,
        )

    @_es_guard
    def _aggregate(self, hits: list[SearchHit]) -> list[DocumentAggregate]:
        grouped: dict[str, list[SearchHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.doc_id, []).append(hit)
        if not grouped:
            return []

        response = self._client.search(
            index=settings.es_index_documents,
            query={"terms": {"doc_id": list(grouped)}},
            size=len(grouped),
        )
        sources: dict[str, dict[str, Any]] = {}
        for document in (response.get("hits") or {}).get("hits") or []:
            source = document.get("_source") or {}
            sources[str(source.get("doc_id"))] = source

        aggregates: list[DocumentAggregate] = []
        for doc_id, matched in grouped.items():
            matched.sort(key=lambda hit: hit.chunk_index)
            source = sources.get(doc_id, {})
            aggregates.append(
                DocumentAggregate(
                    doc_id=doc_id,
                    title=matched[0].title or source.get("title"),
                    best_score=max(hit.score for hit in matched),
                    hit_count=len(matched),
                    chunk_count=int(
                        source.get("chunk_count") or len(matched)
                    ),
                    full_text=str(source.get("full_text") or ""),
                    matched_chunks=matched,
                )
            )

        aggregates.sort(key=lambda doc: doc.best_score, reverse=True)
        return aggregates

    @_es_guard
    def list_documents(self) -> list[DocumentSummary]:
        self._ensure_ready()
        response = self._client.search(
            index=settings.es_index_documents,
            query={"match_all": {}},
            sort=[{"created_at": "desc"}],
            size=settings.knowledge_list_limit,
        )
        summaries: list[DocumentSummary] = []
        for document in (response.get("hits") or {}).get("hits") or []:
            source = document.get("_source") or {}
            summaries.append(
                DocumentSummary(
                    doc_id=str(source.get("doc_id")),
                    title=source.get("title"),
                    chunk_count=int(source.get("chunk_count") or 0),
                    char_count=int(source.get("char_count") or 0),
                )
            )
        return summaries

    @_es_guard
    def get_document(
        self, doc_id: str, include_vectors: bool = False
    ) -> DocumentDetail:
        self._ensure_ready()
        with self._lock:
            if not self._client.exists(
                index=settings.es_index_documents, id=doc_id
            ):
                raise DocumentNotFoundError(doc_id)

            source = self._client.get(
                index=settings.es_index_documents, id=doc_id
            ).get("_source") or {}

            body: dict[str, Any] = {
                "query": {"term": {"doc_id": doc_id}},
                "sort": [{"chunk_index": "asc"}],
                "size": 10000,
            }
            if not include_vectors:
                body["_source"] = {"excludes": ["embedding"]}
            response = self._client.search(
                index=settings.es_index_chunks, **body
            )

            chunks: list[ChunkOut] = []
            for document in (response.get("hits") or {}).get("hits") or []:
                chunk_source = document.get("_source") or {}
                chunks.append(
                    ChunkOut(
                        chunk_id=str(
                            chunk_source.get("chunk_id")
                            or document.get("_id")
                        ),
                        chunk_index=int(
                            chunk_source.get("chunk_index") or 0
                        ),
                        text=str(chunk_source.get("content") or ""),
                        vector=(
                            chunk_source.get("embedding")
                            if include_vectors
                            else None
                        ),
                    )
                )

        return DocumentDetail(
            doc_id=doc_id,
            title=source.get("title"),
            chunk_count=len(chunks),
            full_text=str(source.get("full_text") or ""),
            chunks=chunks,
        )

    @_es_guard
    def delete_document(self, doc_id: str) -> None:
        self._ensure_ready()
        with self._lock:
            if not self._client.exists(
                index=settings.es_index_documents, id=doc_id
            ):
                raise DocumentNotFoundError(doc_id)

            self._client.delete_by_query(
                index=settings.es_index_chunks,
                query={"term": {"doc_id": doc_id}},
                refresh=True,
            )
            self._client.delete(
                index=settings.es_index_documents,
                id=doc_id,
                refresh=True,
                ignore=[404],
            )


_service: KnowledgeService | None = None
_service_lock = threading.Lock()


def get_knowledge_service() -> KnowledgeService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = KnowledgeService()
    return _service
