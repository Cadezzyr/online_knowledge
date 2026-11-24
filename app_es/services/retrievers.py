"""基于 Elasticsearch 的向量检索与关键词检索器。

两者都是同步 `BaseRetriever`，可以直接在 FastAPI 线程池中调用，
并由 `QueryFusionRetriever` 组合成混合检索。
"""

from __future__ import annotations

from typing import Any, Sequence

from elasticsearch import Elasticsearch
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)


def _hit_to_node(hit: dict[str, Any]) -> NodeWithScore:
    source = hit.get("_source") or {}
    doc_id = str(source.get("doc_id") or "")
    node = TextNode(
        text=source.get("content") or "",
        id_=hit.get("_id"),
        metadata={
            "doc_id": doc_id,
            "chunk_index": int(source.get("chunk_index") or 0),
            "title": source.get("title"),
        },
    )
    if doc_id:
        node.relationships = {
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)
        }
    return NodeWithScore(node=node, score=float(hit.get("_score") or 0.0))


class ESVectorRetriever(BaseRetriever):
    """ES kNN 向量召回。"""

    def __init__(
        self,
        client: Elasticsearch,
        index_name: str,
        embed_model: BaseEmbedding,
        similarity_top_k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._index_name = index_name
        self._embed_model = embed_model
        self._k = max(1, similarity_top_k)
        self._doc_ids = list(doc_ids) if doc_ids else None

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        vector = query_bundle.embedding
        if vector is None:
            vector = self._embed_model.get_query_embedding(
                query_bundle.query_str
            )

        knn: dict[str, Any] = {
            "field": "embedding",
            "query_vector": vector,
            "k": self._k,
            "num_candidates": max(self._k * 10, 100),
        }
        if self._doc_ids:
            knn["filter"] = [{"terms": {"doc_id": self._doc_ids}}]

        response = self._client.search(
            index=self._index_name,
            knn=knn,
            size=self._k,
            _source={"excludes": ["embedding"]},
        )
        return [
            _hit_to_node(hit)
            for hit in (response.get("hits") or {}).get("hits") or []
        ]


class ESKeywordRetriever(BaseRetriever):
    """ES BM25 关键词召回。"""

    def __init__(
        self,
        client: Elasticsearch,
        index_name: str,
        similarity_top_k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._index_name = index_name
        self._k = max(1, similarity_top_k)
        self._doc_ids = list(doc_ids) if doc_ids else None

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        bool_query: dict[str, Any] = {
            "must": [
                {
                    "multi_match": {
                        "query": query_bundle.query_str,
                        "fields": ["content", "title^2"],
                        "type": "best_fields",
                    }
                }
            ]
        }
        if self._doc_ids:
            bool_query["filter"] = [{"terms": {"doc_id": self._doc_ids}}]

        response = self._client.search(
            index=self._index_name,
            query={"bool": bool_query},
            size=self._k,
            _source={"excludes": ["embedding"]},
        )
        return [
            _hit_to_node(hit)
            for hit in (response.get("hits") or {}).get("hits") or []
        ]
