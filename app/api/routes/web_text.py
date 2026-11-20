"""web 正文提取相关接口。"""

from fastapi import APIRouter, HTTPException

from app.schemas.web_text import ExtractedText, HtmlTextRequest, UrlTextRequest
from app.services.web_text import (
    ExtractionFailedError,
    FetchFailedError,
    extract_from_html,
    extract_from_url,
)

router = APIRouter(prefix="/web_text", tags=["web_text"])


@router.post("/url_text", response_model=ExtractedText, summary="根据网站 url 提取正文")
def url_text(req: UrlTextRequest) -> ExtractedText:
    """下载 url 对应页面并提取正文与元数据。"""
    try:
        return extract_from_url(str(req.url), req.options)
    except FetchFailedError as exc:
        raise HTTPException(
            status_code=502, detail=f"网页下载失败: {exc}"
        ) from exc
    except ExtractionFailedError:
        raise HTTPException(
            status_code=422, detail="未能从页面中提取正文"
        ) from None


@router.post("/html_text", response_model=ExtractedText, summary="根据 html 提取正文")
def html_text(req: HtmlTextRequest) -> ExtractedText:
    """直接从传入的 HTML 中提取正文与元数据。"""
    try:
        url = str(req.url) if req.url else None
        return extract_from_html(req.html, url, req.options)
    except ExtractionFailedError:
        raise HTTPException(
            status_code=422, detail="未能从页面中提取正文"
        ) from None
