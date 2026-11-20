# web_text 接口规范化改造

## Summary

重构 `app/api/routes/web_text.py` 的两个正文提取接口：GET → POST、`**kwargs` → 显式校验的参数模型、纯字符串返回 → 结构化正文 + 元数据、无错误处理 → 明确的 HTTP 状态码；提取逻辑下沉到 `app/services/web_text.py`，请求/响应模型放到 `app/schemas/web_text.py`。

## Current State Analysis

现有 [web_text.py](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/app/api/routes/web_text.py) 的问题：

1. **导入方式绕过了安装包**：`from sources.trafilatura.trafilatura import ...`，依赖运行时 cwd 在项目根目录，否则 `sources` 命名空间包不可导入；项目已把 trafilatura 以 editable 方式装好，应直接 `from trafilatura import ...`。
2. **两个接口都是 GET**：`html_text` 把整段 HTML 放在 query 参数里，长文档会超出 URL 长度限制。
3. **参数不受控**：`**kwargs` 直接把任意 query 参数透传给 `extract`，既无校验也污染 OpenAPI 文档。
4. **返回裸字符串**：丢弃了 trafilatura 已提取的 title/author/date/source 等元数据，调用方拿不到出处信息。
5. **无错误处理**：`fetch_url` 失败返回 `None`，`extract` 失败也返回 `None`，当前代码会对 `None` 调 `json.loads` 直接抛 500。
6. **`html_text` 逻辑有 bug**：`output_format="html"` 返回的是 HTML 字符串，并非 JSON，`json.loads(all_data)` 必然失败。
7. 路由缺少 `tags`，在 `/docs` 里没有分组。

可复用的项目约定（来自已完成的 knowledge 模块）：
- [knowledge.py 路由](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/app/api/routes/knowledge.py)：同步 `def`（FastAPI 走线程池）、`response_model`、`APIRouter(prefix=..., tags=[...])`、异常映射为 `HTTPException`。
- [knowledge.py 服务](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/app/services/knowledge.py)：自定义异常 + 单例获取函数。
- [knowledge.py 模型](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/app/schemas/knowledge.py)：pydantic v2 `BaseModel` + `Field` 约束。

已确认的 trafilatura API（2.3.1，源码在 `sources/trafilatura`）：
- `fetch_url(url, no_ssl=False, ...) -> str | None`，失败返回 `None`。
- `extract(filecontent, url=None, output_format="json", with_metadata=True, include_comments=..., include_tables=..., include_images=..., include_links=..., favor_precision=..., favor_recall=..., deduplicate=..., target_language=...) -> str | None`。
- `output_format="json"` + `with_metadata=True` 的字段包含 `title`、`author`、`source`(URL)、`hostname`、`source-hostname`(站点名)、`excerpt`(描述)、`date`、`categories`、`tags`、`language`、`text`、`comments`；其中 `categories` / `tags` 是分号拼接的字符串。
- 抓取层自带 `_vet_peer`，拒绝非公网地址，可缓解 SSRF。

## Proposed Changes

### 1. 新增 `app/schemas/web_text.py`

```python
class ExtractOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_comments: bool | None = None
    include_tables: bool | None = None
    include_images: bool | None = None
    include_links: bool | None = None
    favor_precision: bool | None = None
    favor_recall: bool | None = None
    deduplicate: bool | None = None
    target_language: str | None = Field(default=None, min_length=2, max_length=5)
```

全部默认 `None`，服务层用 `exclude_unset=True` 只透传调用方显式给出的项，未传时沿用 trafilatura 自身默认值。

```python
class UrlTextRequest(BaseModel):
    url: HttpUrl
    options: ExtractOptions = Field(default_factory=ExtractOptions)

class HtmlTextRequest(BaseModel):
    html: str = Field(min_length=1)
    url: HttpUrl | None = None      # 可选，仅用于补全 source 等元数据
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
```

`HttpUrl` 限定 http/https，拒绝 `file://` 等协议。

### 2. 新增 `app/services/web_text.py`

自定义异常：`FetchFailedError`（下载失败）、`ExtractionFailedError`（无正文或结果不可解析）。

```python
from trafilatura import extract, fetch_url

def extract_from_url(url: str, options: ExtractOptions) -> ExtractedText:
    html = fetch_url(url)
    if not html:
        raise FetchFailedError(url)
    return _extract(html, url, options)

def extract_from_html(html: str, url: str | None, options: ExtractOptions) -> ExtractedText:
    return _extract(html, url, options)
```

`_extract` 要点：

- 组装参数：`output_format="json"`、`with_metadata=True`；`url` 非空时传入；其余字段来自 `options.model_dump(exclude_unset=True, exclude_none=True)`。
- `raw = extract(html, **kwargs)`；`raw` 为 `None` → `ExtractionFailedError`；`json.loads(raw)` 抛异常 → 同样转为 `ExtractionFailedError`。
- 字段映射：`source → url`、`source-hostname → sitename`、`excerpt → description`；`categories` / `tags` 按 `;` 切分并去空。
- 校验正文：`text` 为空字符串 → `ExtractionFailedError`。
- 统计：`char_count = len(text)`、`paragraph_count = len([p for p in text.split("\n") if p.strip()])`。
- 返回 `ExtractedText`，缺失字段一律 `None` / 空列表。

### 3. 重写 `app/api/routes/web_text.py`

```python
router = APIRouter(prefix="/web_text", tags=["web_text"])
```

| 方法 | 路径 | 入参 | 说明 |
| --- | --- | --- | --- |
| POST | `/web_text/url_text` | `UrlTextRequest` | 下载网页并提取正文，返回 `ExtractedText` |
| POST | `/web_text/html_text` | `HtmlTextRequest` | 直接提取传入 HTML 的正文，返回 `ExtractedText` |

两个处理函数均为同步 `def`，`response_model=ExtractedText`，异常映射：

- `FetchFailedError` → `502`，`detail="网页下载失败: {url}"`
- `ExtractionFailedError` → `422`，`detail="未能从页面中提取正文"`

原 GET 版本直接删除（项目内无其他调用方，`app/api/router.py` 只引用了 `router`，无需改动）。

### 4. 其他

- 服务层不再 import `sources.trafilatura...`，改用已安装的 `trafilatura` 包；这样即使从非项目根目录启动 uvicorn 也不会导入失败。
- 不新增依赖、不修改 `pyproject.toml`。

## Assumptions & Decisions

- 路径保持 `/web_text/url_text`、`/web_text/html_text`，仅把方法改为 POST（用户已确认）。
- 返回结构化正文 + 元数据（用户已确认）；`html_text` 统一用 `output_format="json"` 提取纯文本，修复原有的 `json.loads(html 输出)` bug。
- 错误统一用 HTTP 状态码 + `detail`（用户已确认），不引入业务码包装。
- 不暴露 `output_format` / `only_with_metadata` / `settingsfile` 等高级参数，保持接口面最小；如后续需要再扩展 `ExtractOptions`。
- 不设置自定义超时（trafilatura `fetch_url` 未开放该参数），超时行为沿用其内部配置。

## Verification

1. **导入与路由**
   ```powershell
   .\.venv\Scripts\python.exe -c "from app.main import app; print([p for p in app.openapi()['paths'] if 'web_text' in p])"
   ```
   预期只出现两个路径，且 openapi 中方法为 `post`。

2. **html 提取（不依赖网络）** — TestClient 提交带 `og:title`、`author`、`date` meta 的中文文章 HTML：
   - 断言 `200`、`title` 非空、`text` 包含正文段落、`char_count > 0`、`paragraph_count >= 2`。
   - 断言 `categories` / `tags` 为列表（即便为空列表）。

3. **错误分支**
   - `html=""` → `422`（pydantic 校验）。
   - 传入无正文 HTML（如 `<html><body></body></html>`）→ `422`、`detail="未能从页面中提取正文"`。
   - `url="file:///C:/x.html"` → `422`（`HttpUrl` 校验）。

4. **url 提取** — 用 monkeypatch 替换 `app.services.web_text.fetch_url`：
   - 返回样例 HTML → `200` 且字段与样例一致；
   - 返回 `None` → `502`、`detail` 含 URL。
   如外网可用，再补一次真实请求（如 `https://example.com`）确认端到端。

5. **文档**：访问 `/docs`，确认出现 `web_text` 分组，两个接口都有请求体示例与响应模型。
