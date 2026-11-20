"""llama_index 分片知识库接口：入库、片段召回、文档聚合。"""

from fastapi import APIRouter, HTTPException, Query

from app.schemas.knowledge import (
    DeleteResponse,
    DocumentDetail,
    DocumentSummary,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.knowledge import (
    DocumentExistsError,
    DocumentNotFoundError,
    RerankerUnavailableError,
    get_knowledge_service,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/documents", response_model=IngestResponse, summary="长文入库并按段落切片")
def ingest_document(req: IngestRequest) -> IngestResponse:
    try:
        return get_knowledge_service().ingest(req)
    except DocumentExistsError as exc:
        raise HTTPException(status_code=409, detail=f"doc_id 已存在: {exc}") from exc


@router.post("/search", response_model=SearchResponse, summary="按片段召回并聚合回整篇")
def search_knowledge(req: SearchRequest) -> SearchResponse:
    try:
        return get_knowledge_service().search(req)
    except RerankerUnavailableError as exc:
        raise HTTPException(
            status_code=503, detail=f"精排模型未就绪: {exc}"
        ) from exc


@router.get("/documents", response_model=list[DocumentSummary], summary="列出已入库文档")
def list_documents() -> list[DocumentSummary]:
    return get_knowledge_service().list_documents()


@router.get("/documents/{doc_id}", response_model=DocumentDetail, summary="查看文档与分片")
def get_document(
    doc_id: str,
    include_vectors: bool = Query(default=False, description="是否返回每个分片的向量"),
) -> DocumentDetail:
    try:
        return get_knowledge_service().get_document(doc_id, include_vectors)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {exc}") from exc


@router.delete("/documents/{doc_id}", response_model=DeleteResponse, summary="删除文档及其分片")
def delete_document(doc_id: str) -> DeleteResponse:
    try:
        get_knowledge_service().delete_document(doc_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"文档不存在: {exc}") from exc
    return DeleteResponse(doc_id=doc_id, deleted=True)
