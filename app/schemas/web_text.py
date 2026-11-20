"""web 正文提取的请求/响应模型。"""

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ExtractOptions(BaseModel):
    """可选提取参数；未显式传入时沿用 trafilatura 自身默认值。"""

    model_config = ConfigDict(extra="forbid")

    include_comments: bool | None = None
    include_tables: bool | None = None
    include_images: bool | None = None
    include_links: bool | None = None
    favor_precision: bool | None = None
    favor_recall: bool | None = None
    deduplicate: bool | None = None
    target_language: str | None = Field(default=None, min_length=2, max_length=5)


class UrlTextRequest(BaseModel):
    url: HttpUrl
    options: ExtractOptions = Field(default_factory=ExtractOptions)


class HtmlTextRequest(BaseModel):
    html: str = Field(min_length=1)
    url: HttpUrl | None = None  # 可选，仅用于补全 source 等元数据
    options: ExtractOptions = Field(default_factory=ExtractOptions)


class ExtractedText(BaseModel):
    url: str | None
    title: str | None
    author: str | None
    date: str | None
    sitename: str | None
    description: str | None
    categories: list[str]
    tags: list[str]
    language: str | None
    text: str
    comments: str | None
    char_count: int
    paragraph_count: int
