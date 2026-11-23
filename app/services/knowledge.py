"""基于 llama_index 的分片索引服务。

每个 chunk 的文本存 docstore，向量存 SimpleVectorStore；检索按片段召回后按文档聚合。
"""

from __future__ import annotations

import threading
import os
from pathlib import Path
from typing import cast
from uuid import uuid4

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.utils import split_by_regex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever


from app.core.config import settings
from app.schemas.knowledge import (
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


class DocumentExistsError(Exception):
    """doc_id 已存在。"""


class DocumentNotFoundError(Exception):
    """doc_id 不存在。"""


class RerankerUnavailableError(Exception):
    """精排模型未就绪（模型目录缺失或加载失败）。"""


def _char_tokenizer(text: str) -> list[str]:
    """按字符计数，避免 SentenceSplitter 默认 tokenizer 联网下载编码表。"""
    return list(text)


_SENTENCE_REGEX = r"[^,.;。？！]+[,.;。？！]?|[,.;。？！]"


class KnowledgeService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._persist_dir = Path(settings.knowledge_store_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            embed_batch_size=settings.embedding_batch_size,
        )
        self._index = self._load_or_create_v_index()
        # BM25 索引按需构建；入库或删除后置空，下次检索时重建
        self._bm25_retriever: BM25Retriever | None = None
        # 精排模型按需加载，首次请求 rerank=true 时才占用内存
        self._reranker: SentenceTransformerRerank | None = None
        # 项目暂不接入真实 LLM。llama_index 的合成器等组件在构造时会读取
        # 全局 Settings.llm，这里统一指向 MockLLM，避免依赖 LLM 供应商包。
        Settings.llm = MockLLM()

    def _load_or_create_v_index(self) -> VectorStoreIndex:
        if (self._persist_dir / "index_store.json").exists():
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self._persist_dir)
            )
            return cast(
                VectorStoreIndex,
                load_index_from_storage(
                    storage_context, embed_model=self._embed_model
                ),
            )

        storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
        return VectorStoreIndex(
            [], storage_context=storage_context, embed_model=self._embed_model
        )

    def _persist(self) -> None:
        self._index.storage_context.persist(persist_dir=str(self._persist_dir))

    def _chunk_nodes(self) -> list[BaseNode]:
        """只取分片节点，排除入库时单独保存的整篇原文。"""
        nodes: list[BaseNode] = []
        ref_infos = self._index.docstore.get_all_ref_doc_info() or {}
        for ref_info in ref_infos.values():
            for node_id in ref_info.node_ids:
                node = self._index.docstore.get_document(node_id, raise_error=False)
                if node is not None:
                    nodes.append(node)
        return nodes

    def _get_bm25_retriever(
        self, nodes: list[BaseNode], top_k: int
    ) -> BM25Retriever:
        """复用已建好的 BM25 索引，只在首次或数据变更后重建"""
        # bm25s 要求 k 不超过语料规模，否则检索时报错
        k = max(1, min(top_k, len(nodes)))
        # 缓存实例可能是语料更少时建的，与当前语料不一致就重建
        if self._bm25_retriever is not None:
            indexed = int(self._bm25_retriever.bm25.scores.get("num_docs") or 0)
            if indexed != len(nodes):
                self._bm25_retriever = None
        if self._bm25_retriever is None:
            # if os.path.isdir('D:/Pycharm Projects/联网知识库情况/online_knowledge/data/knowledge/bm25_retriever'):
            #     self._bm25_retriever = BM25Retriever.from_persist_dir("D:/Pycharm Projects/联网知识库情况/online_knowledge/data/knowledge/bm25_retriever")
            #     return self._bm25_retriever
            self._bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=k,
                language="zh",
                # 默认 stemmer 是英文词干化，对中文无意义
                skip_stemming=True,
                # 中文没有空格，默认正则会整句算一个 token，这里按单字切
                token_pattern=r"(?u)[\w]",
            )
        else:
            # 构造期只会夹一次，复用缓存时按索引真实规模再兜底
            indexed = int(self._bm25_retriever.bm25.scores.get("num_docs") or 0)
            self._bm25_retriever.similarity_top_k = max(1, min(k, indexed or k))
        return self._bm25_retriever

    def _make_retriever(self, req: SearchRequest) -> BaseRetriever:
        """纯向量召回，或向量 + BM25 两路融合。"""
        # 候选池要给 rerank 留素材，先放宽到 top_k 的 4 倍（至少 20）
        candidates = max(req.top_k * 4, 20)
        if req.mode != "hybrid":
            return self._index.as_retriever(similarity_top_k=candidates)

        nodes = self._chunk_nodes()
        if not nodes:  # 空库时 BM25 无法建索引，退化为纯向量
            return self._index.as_retriever(similarity_top_k=candidates)

        # 候选池不能超过语料规模：bm25s 要求 k <= 索引里的文档数
        candidates = min(candidates, len(nodes))
        return QueryFusionRetriever(
            [
                self._index.as_retriever(similarity_top_k=candidates),
                self._get_bm25_retriever(nodes, candidates),
            ],
            # 构造函数会读取 Settings.llm，传个空壳避免要求安装 LLM 包；
            # num_queries=1 时它不会被调用
            llm=MockLLM(),
            # 融合后保留候选池供后处理（如 rerank）精排，最终截断交给 search()
            similarity_top_k=candidates,
            num_queries=1,  # 不启用 LLM 查询改写
            # RRF 不读取 retriever_weights，要用 alpha 必须选相对分数融合
            mode=FUSION_MODES.RELATIVE_SCORE,
            use_async=False,
            retriever_weights=[req.alpha, 1.0 - req.alpha],
        )

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
        """后处理链：rerank、阈值过滤、去重都往这里加，search() 无需改动。"""
        if not req.rerank:
            return []
        return [self._get_reranker()]

    def _build_engine(self, req: SearchRequest) -> RetrieverQueryEngine:
        """把召回与后处理装成统一管线。

        NO_TEXT 表示只取节点、不合成答案；该模式下 LLM 不会被调用，
        仅为满足合成器构造签名而使用 MockLLM。
        """
        return RetrieverQueryEngine.from_args(
            retriever=self._make_retriever(req),
            llm=MockLLM(),
            response_mode=ResponseMode.NO_TEXT,
            node_postprocessors=self._make_postprocessors(req),
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

    def ingest(self, req: IngestRequest) -> IngestResponse:
        doc_id = req.doc_id or str(uuid4())
        splitter = self._make_splitter(req.chunk_size, req.chunk_overlap)

        with self._lock:
            if self._index.docstore.document_exists(doc_id):
                raise DocumentExistsError(doc_id)

            metadata = {"doc_id": doc_id}
            if req.title is not None:
                metadata["title"] = req.title
            document = Document(text=req.text, id_=doc_id, metadata=metadata)

            nodes = splitter.get_nodes_from_documents([document])
            for index, node in enumerate(nodes):
                node.metadata["chunk_index"] = index

            # 原文单独入库，供检索聚合时还原整篇
            self._index.docstore.add_documents([document])
            # 分片文本落 docstore，向量落 SimpleVectorStore
            self._index.insert_nodes(nodes)
            self._persist()
            # 语料变了，BM25 索引失效
            self._bm25_retriever = None

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

    def search(self, req: SearchRequest) -> SearchResponse:
        with self._lock:
            engine = self._build_engine(req)
            # 检索层返回候选池，这里统一截到 top_k；接入 rerank 后由其完成精排截断
            results: list[NodeWithScore] = engine.retrieve(
                QueryBundle(req.query)
            )[: req.top_k]

            hits: list[SearchHit] = []
            for result in results:
                node = result.node
                doc_id = node.ref_doc_id or node.metadata.get("doc_id")
                if doc_id is None:
                    continue
                if req.doc_ids and doc_id not in req.doc_ids:
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

    def _aggregate(self, hits: list[SearchHit]) -> list[DocumentAggregate]:
        grouped: dict[str, list[SearchHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.doc_id, []).append(hit)

        aggregates: list[DocumentAggregate] = []
        for doc_id, matched in grouped.items():
            matched.sort(key=lambda hit: hit.chunk_index)
            source = self._index.docstore.get_document(doc_id, raise_error=False)
            ref_info = self._index.docstore.get_ref_doc_info(doc_id)
            aggregates.append(
                DocumentAggregate(
                    doc_id=doc_id,
                    title=matched[0].title,
                    best_score=max(hit.score for hit in matched),
                    hit_count=len(matched),
                    chunk_count=len(ref_info.node_ids) if ref_info else len(matched),
                    full_text=getattr(source, "text", "") if source else "",
                    matched_chunks=matched,
                )
            )

        aggregates.sort(key=lambda doc: doc.best_score, reverse=True)
        return aggregates

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock:
            ref_infos = self._index.docstore.get_all_ref_doc_info() or {}
            summaries: list[DocumentSummary] = []
            for doc_id, ref_info in ref_infos.items():
                source = self._index.docstore.get_document(doc_id, raise_error=False)
                summaries.append(
                    DocumentSummary(
                        doc_id=doc_id,
                        title=source.metadata.get("title") if source else None,
                        chunk_count=len(ref_info.node_ids),
                        char_count=len(getattr(source, "text", "") or ""),
                    )
                )
        return summaries

    def get_document(self, doc_id: str, include_vectors: bool = False) -> DocumentDetail:
        with self._lock:
            source = self._index.docstore.get_document(doc_id, raise_error=False)
            ref_info = self._index.docstore.get_ref_doc_info(doc_id)
            if source is None and ref_info is None:
                raise DocumentNotFoundError(doc_id)

            chunks: list[ChunkOut] = []
            if ref_info is not None:
                for node_id in ref_info.node_ids:
                    node = self._index.docstore.get_document(node_id, raise_error=False)
                    if node is None:
                        continue
                    chunks.append(
                        ChunkOut(
                            chunk_id=node_id,
                            chunk_index=int(node.metadata.get("chunk_index", 0)),
                            text=node.get_content(),
                            vector=(
                                self._index.vector_store.get(node_id)
                                if include_vectors
                                else None
                            ),
                        )
                    )
                chunks.sort(key=lambda chunk: chunk.chunk_index)

        return DocumentDetail(
            doc_id=doc_id,
            title=source.metadata.get("title") if source else None,
            chunk_count=len(chunks),
            full_text=getattr(source, "text", "") if source else "",
            chunks=chunks,
        )

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            source = self._index.docstore.get_document(doc_id, raise_error=False)
            ref_info = self._index.docstore.get_ref_doc_info(doc_id)
            if source is None and ref_info is None:
                raise DocumentNotFoundError(doc_id)

            if ref_info is not None:
                self._index.delete_ref_doc(doc_id, delete_from_docstore=True)
            else:
                self._index.docstore.delete_document(doc_id)
            self._persist()
            # 语料变了，BM25 索引失效
            self._bm25_retriever = None


_service: KnowledgeService | None = None
_service_lock = threading.Lock()


def get_knowledge_service() -> KnowledgeService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = KnowledgeService()
    return _service
