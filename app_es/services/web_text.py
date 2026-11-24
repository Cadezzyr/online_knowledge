"""基于 trafilatura 的网页正文提取服务。"""

from __future__ import annotations

import json

from trafilatura import extract, fetch_url

from app_es.schemas.web_text import ExtractOptions, ExtractedText


class FetchFailedError(Exception):
    """网页下载失败。"""


class ExtractionFailedError(Exception):
    """未能从页面中提取正文。"""


def _split_joined(value: object) -> list[str]:
    """trafilatura 的 categories/tags 是分号拼接字符串。"""
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _extract(html: str, url: str | None, options: ExtractOptions) -> ExtractedText:
    params = options.model_dump(exclude_unset=True, exclude_none=True)
    kwargs: dict[str, object] = {
        "output_format": "json",
        "with_metadata": True,
        **params,
    }
    if url:
        kwargs["url"] = url

    raw = extract(html, **kwargs)
    if not raw:
        raise ExtractionFailedError("empty extraction result")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionFailedError("invalid extraction result") from exc

    text = (data.get("text") or "").strip()
    if not text:
        raise ExtractionFailedError("no content extracted")

    return ExtractedText(
        url=data.get("source") or url,
        title=data.get("title"),
        author=data.get("author"),
        date=data.get("date"),
        sitename=data.get("source-hostname") or data.get("hostname"),
        description=data.get("excerpt"),
        categories=_split_joined(data.get("categories")),
        tags=_split_joined(data.get("tags")),
        language=data.get("language"),
        text=text,
        comments=data.get("comments") or None,
        char_count=len(text),
        paragraph_count=len([line for line in text.split("\n") if line.strip()]),
    )


def extract_from_url(url: str, options: ExtractOptions) -> ExtractedText:
    html = fetch_url(url)
    if not html:
        raise FetchFailedError(url)
    return _extract(html, url, options)


def extract_from_html(
    html: str, url: str | None, options: ExtractOptions
) -> ExtractedText:
    return _extract(html, url, options)
