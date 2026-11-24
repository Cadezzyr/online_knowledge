"""Elasticsearch 客户端、索引映射与健康检查。"""

from __future__ import annotations

import logging
import threading
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError
from elasticsearch.exceptions import TransportError

from app_es.core.config import settings

logger = logging.getLogger(__name__)


class ElasticsearchUnavailableError(Exception):
    """Elasticsearch 连接失败或不可用。"""


_client: Elasticsearch | None = None
_client_lock = threading.Lock()


def get_es_client() -> Elasticsearch:
    """进程内共享的同步 ES 客户端（懒创建）。"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                auth: dict[str, Any] = {}
                if settings.es_api_key:
                    auth["api_key"] = settings.es_api_key
                elif settings.es_username and settings.es_password:
                    auth["basic_auth"] = (
                        settings.es_username,
                        settings.es_password,
                    )
                _client = Elasticsearch(
                    hosts=[settings.es_url],
                    verify_certs=settings.es_verify_certs,
                    ca_certs=settings.es_ca_certs,
                    request_timeout=settings.es_request_timeout,
                    **auth,
                )
    return _client


def _text_field_mapping() -> dict[str, Any]:
    return {
        "type": "text",
        "analyzer": settings.es_analyzer,
        "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
    }


def _chunks_mapping() -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": settings.es_number_of_shards,
            "number_of_replicas": settings.es_number_of_replicas,
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "ref_doc_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "title": _text_field_mapping(),
                "content": {"type": "text", "analyzer": settings.es_analyzer},
                "embedding": {
                    "type": "dense_vector",
                    "dims": settings.es_embedding_dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "created_at": {"type": "date"},
            }
        },
    }


def _documents_mapping() -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": settings.es_number_of_shards,
            "number_of_replicas": settings.es_number_of_replicas,
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": _text_field_mapping(),
                "full_text": {"type": "text", "analyzer": settings.es_analyzer},
                "char_count": {"type": "integer"},
                "chunk_count": {"type": "integer"},
                "created_at": {"type": "date"},
            }
        },
    }


def ensure_indices(client: Elasticsearch | None = None) -> None:
    """幂等创建文档索引与分片索引；已存在则不动。"""
    client = client or get_es_client()
    try:
        for name, body in (
            (settings.es_index_documents, _documents_mapping()),
            (settings.es_index_chunks, _chunks_mapping()),
        ):
            if not client.indices.exists(index=name):
                client.indices.create(index=name, **body)
    except (ESConnectionError, TransportError) as exc:
        raise ElasticsearchUnavailableError(str(exc)) from exc


def close_es_client() -> None:
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            finally:
                _client = None


def es_health() -> dict[str, Any]:
    """返回连接与索引状态，不抛异常（供探活接口使用）。"""
    client = get_es_client()
    try:
        info = client.info()
        return {
            "status": "ok",
            "url": settings.es_url,
            "cluster_name": info.get("cluster_name"),
            "version": (info.get("version") or {}).get("number"),
            "documents_index": bool(
                client.indices.exists(index=settings.es_index_documents)
            ),
            "chunks_index": bool(
                client.indices.exists(index=settings.es_index_chunks)
            ),
        }
    except (ESConnectionError, TransportError) as exc:
        logger.warning("Elasticsearch 健康检查失败: %s", exc)
        return {
            "status": "unavailable",
            "url": settings.es_url,
            "cluster_name": None,
            "version": None,
            "documents_index": False,
            "chunks_index": False,
        }
