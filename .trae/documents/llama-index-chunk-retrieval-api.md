# 基于 llama_index 的分片存储与检索接口

## Summary

在 `app/api/routes/` 新增知识入库 / 检索接口，使用 `sources/llama_index` 中的 `llama-index-core` 完成：
长正文按段落切分为多个 chunk，**每个 chunk 独立保存文本与向量**；检索时**按 chunk 召回**，再按文档聚合、还原整篇正文。

仅提供后端接口，不做前端与鉴权。向量化使用本地 `sentence-transformers` 模型（用户已选），索引持久化到磁盘。

## Current State Analysis

- 应用入口 [app/main.py](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/app/main.py)：FastAPI + lifespan（当前为空），已挂载 `api_router`，前缀 `/api/v1`。
- [app/api/router.py](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/app/api/router.py#L1-L6)：`api_router` 只 include 了 `web_text`，注释里给出了后续注册示例。
- 现有路由 [web_text.py](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/app/api/routes/web_text.py)（trafilatura 版）与 [retrieve.py](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/app/api/routes/retrieve.py) 内容重复；`retrieve.py` 未被注册且 `from sources.llama_index import llama` 是无效导入。本方案不改动这两个文件。
- 无 `schemas` / `services` 分层，当前接口直接在路由里写逻辑。本方案新增这两层，避免把索引逻辑堆在路由里。
- [app/core/config.py](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/app/core/config.py)：`pydantic-settings`，读取根目录 `.env`，现有字段 `app_name` / `api_v1_prefix` / `debug`。
- [pyproject.toml](file:///d:/Pycharm%20Projects/联网知识库情况/online_knowledge/pyproject.toml)：`sources/llama_index/llama-index-core` 已通过 `[tool.uv.sources]` 以 editable 方式声明；当前**未安装任何 embedding 集成包**，`llama-index-core` 中只有 `MockEmbedding`（返回常量向量，检索无区分度，不可用于验证召回）。
- 已确认可用的 llama_index API：
  - `llama_index.core.node_parser.SentenceSplitter`（支持 `paragraph_separator`、`tokenizer` 参数）
  - `llama_index.core.vector_stores.SimpleVectorStore`（`stores_text=False`，节点文本会落到 docstore）
  - `llama_index.core.storage.StorageContext.persist()` / `load_index_from_storage()`
  - `VectorStoreIndex.insert_nodes()` / `delete_ref_doc(ref_doc_id, delete_from_docstore=True)`
  - `SimpleDocumentStore.get_document()` / `get_ref_doc_info()` / `get_all_ref_doc_info()`

## Proposed Changes

### 1. 依赖：安装本地 embedding 集成

`pyproject.toml` 的 `dependencies` 增加：

```toml
"llama-index-embeddings-huggingface",
```

安装命令（优先 uv；不可用则退回 pip + 清华镜像）：

```powershell
uv add llama-index-embeddings-huggingface
# 或
.\.venv\Scripts\python.exe -m pip install llama-index-embeddings-huggingface -i https://pypi.tuna.tsinghua.edu.cn/simple
```

说明：该包会带入 `sentence-transformers`、`transformers`、`torch`（CPU wheel，数百 MB）。首次请求时从 HuggingFace 下载模型 `BAAI/bge-small-zh-v1.5`（约 100MB）；若网络不通，设置 `HF_ENDPOINT=https://hf-mirror.com`。

### 2. 配置扩展 — `app/core/config.py`

新增字段（均带默认值，`.env` 可覆盖）：

```python
    # 知识库切片与检索
    knowledge_store_dir: str = "data/knowledge"
    knowledge_chunk_size: int = 512        # 以字符计（见自定义 tokenizer）
    knowledge_chunk_overlap: int = 64
    knowledge_top_k: int = 5
    # 向量化
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
```

`.env.example` 同步追加以上 7 个键（值与默认值一致）。

同时 `.gitignore` 追加 `data/`（索引持久化产物不入库）。

### 3. 请求/响应模型 — 新建 `app/schemas/knowledge.py`（及空的 `app/schemas/__init__.py`）

```python
class ChunkOut(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    vector: list[float] | None = None      # 仅 include_vectors=true 时返回

class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    title: str | None = None
    doc_id: str | None = None               # 不传则自动生成 uuid4
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
    doc_ids: list[str] | None = None        # 召回后按文档过滤
    include_text: bool = True

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
    full_text: str                          # 由源文档还原的整篇正文
    matched_chunks: list[SearchHit]

class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[SearchHit]                   # 片段级召回
    documents: list[DocumentAggregate]      # 按文档聚合

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
```

### 4. 索引服务 — 新建 `app/services/knowledge.py`（及空的 `app/services/__init__.py`）

自定义异常：`DocumentExistsError`、`DocumentNotFoundError`。

`KnowledgeService` 职责与实现要点：

1. **初始化（懒加载单例）**
   - 模块级 `_service` + `threading.Lock` + `get_knowledge_service()`。
   - `HuggingFaceEmbedding(model_name=settings.embedding_model, device=settings.embedding_device, embed_batch_size=settings.embedding_batch_size)`。
   - `persist_dir = Path(settings.knowledge_store_dir)`，`mkdir(parents=True, exist_ok=True)`。
   - `persist_dir / "index_store.json"` 存在 → `StorageContext.from_defaults(persist_dir=...)` + `load_index_from_storage(ctx, embed_model=...)`；否则 `StorageContext.from_defaults(vector_store=SimpleVectorStore())` + `VectorStoreIndex([], storage_context=ctx, embed_model=...)`。
   - 实例内 `threading.RLock` 保护所有读写 + persist。接口全部用同步 `def`（FastAPI 走线程池），避免 CPU 绑定阻塞事件循环。

2. **切片 tokenizer**
   ```python
   def _char_tokenizer(text: str) -> list[str]:
       return list(text)
   ```
   传给 `SentenceSplitter(tokenizer=_char_tokenizer, ...)`，避免连接 tiktoken 在线下载编码表；此时 `chunk_size` 语义为字符数。

3. **`ingest(req)`**
   - `doc_id = req.doc_id or uuid4()`；若 `docstore.document_exists(doc_id)` → 抛 `DocumentExistsError`。
   - `Document(text=req.text, id_=doc_id, metadata={"doc_id": doc_id, "title": req.title})`。
   - `SentenceSplitter(chunk_size=req.chunk_size or settings.knowledge_chunk_size, chunk_overlap=..., paragraph_separator="\n\n", tokenizer=_char_tokenizer)`，`get_nodes_from_documents([document])`。
   - 逐节点写入 `node.metadata["chunk_index"] = i`。
   - 事务内：`index.docstore.add_documents([document])` 保存整篇原文（用于聚合还原）；`index.insert_nodes(nodes)` 完成分片文本落 docstore + 向量落 SimpleVectorStore；`storage_context.persist(persist_dir)`。
   - 返回 `doc_id/title/chunk_count/chunks`（含每片文本，便于核对）。

4. **`search(req)`**
   - `index.as_retriever(similarity_top_k=req.top_k).retrieve(req.query)` → `NodeWithScore` 列表。
   - 映射为 `SearchHit`：`doc_id = node.ref_doc_id or metadata["doc_id"]`，`chunk_index = node.metadata["chunk_index"]`，`score = score or 0.0`。
   - `req.doc_ids` 非空时按 `doc_id` 过滤。
   - 聚合：按 `doc_id` 分组，每组 `best_score=max(score)`、`hit_count=len(chunks)`、`matched_chunks` 按 `chunk_index` 升序；`full_text = docstore.get_document(doc_id).text`；`chunk_count = len(docstore.get_ref_doc_info(doc_id).node_ids)`。`documents` 按 `best_score` 降序。

5. **`list_documents()`**：遍历 `docstore.get_all_ref_doc_info()`，用 `get_document(doc_id)` 取标题与整篇长度；`chunk_count = len(ref_info.node_ids)`。

6. **`get_document(doc_id, include_vectors=False)`**：源文档不存在 → `DocumentNotFoundError`；chunk 列表来自 `ref_info.node_ids` → `docstore.get_document(node_id)`，按 `chunk_index` 排序；`include_vectors` 时用 `index.vector_store.get(node_id)` 取向量。

7. **`delete_document(doc_id)`**：不存在 → `DocumentNotFoundError`；`index.delete_ref_doc(doc_id, delete_from_docstore=True)` 同时删除向量、分片文本和整篇原文；再 `persist`。

### 5. 路由 — 新建 `app/api/routes/knowledge.py`

`router = APIRouter(prefix="/knowledge", tags=["knowledge"])`，全部同步 `def`：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/documents` | 入库并切片，返回 `IngestResponse`；已存在 → 409 |
| POST | `/search` | 片段召回 + 文档聚合，返回 `SearchResponse` |
| GET | `/documents` | 列出文档概要 `list[DocumentSummary]` |
| GET | `/documents/{doc_id}` | 文档详情，query 参数 `include_vectors=false`；不存在 → 404 |
| DELETE | `/documents/{doc_id}` | 删除文档及其全部分片；不存在 → 404 |

异常映射：`DocumentExistsError → HTTPException(409)`，`DocumentNotFoundError → HTTPException(404)`。

### 6. 注册路由 — `app/api/router.py`

```python
from app.api.routes import knowledge, web_text

api_router = APIRouter()
api_router.include_router(web_text.router)
api_router.include_router(knowledge.router)
```

## Assumptions & Decisions

- **向量化**：按用户选择使用本地 `sentence-transformers`（经 `HuggingFaceEmbedding`），模型 `BAAI/bge-small-zh-v1.5`（中文、512 维、体积小）。设备默认 CPU，可用 `EMBEDDING_DEVICE` 覆盖。
- **存储**：按用户选择持久化到 `data/knowledge/`（`docstore.json` 存分片文本与原文，`default__vector_store.json` 存向量，`index_store.json` 存索引结构）。`data/` 加入 `.gitignore`。
- **切片语义**：按段落 `\n\n` 优先切分，段落过长再按句子切，`chunk_size` 以字符计（自定义 tokenizer，规避 tiktoken 在线下载）。默认 512 / overlap 64。
- **"聚合回整篇"**：入库时把原始 `Document` 一并存进 docstore（id=doc_id），检索聚合时直接读取，避免用有重叠的分片拼接导致正文重复。
- **不加鉴权、不做异步、不引入数据库**，符合"测试用最小后端接口"。
- 不改动已有 `web_text.py` / `retrieve.py`（后者本就未注册且导入有误，属既有问题，不在本次范围）。

## Verification

1. **导入与启动**
   ```powershell
   .\.venv\Scripts\python.exe -c "from app.main import app; print([r.path for r in app.routes if '/knowledge' in getattr(r,'path','')])"
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
   ```
   预期输出 5 条 `/api/v1/knowledge/...` 路径；`/docs` 中出现 knowledge 分组。

2. **入库 + 切片**
   ```powershell
   $body = @{ title="测试长文"; text=(("第一段..."*40)+"`n`n"+("第二段..."*40)+"`n`n"+("第三段...关键词..."*40)); chunk_size=200 } | ConvertTo-Json
   $doc = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/knowledge/documents -ContentType application/json -Body $body
   $doc.chunk_count   # 期望 > 1
   ```

3. **片段召回 + 聚合**
   ```powershell
   $q = @{ query="关键词"; top_k=3 } | ConvertTo-Json
   $res = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/knowledge/search -ContentType application/json -Body $q
   $res.hits.Count; $res.documents[0].full_text -eq $bodyText   # 期望 true
   ```
   校验：`hits` 有分数且按分排序；`documents[0].matched_chunks` 的 `chunk_index` 升序。

4. **详情 / 列表 / 向量可查**
   - `GET /api/v1/knowledge/documents/{doc_id}?include_vectors=true` → `chunks[i].vector` 长度为 512。
   - `GET /api/v1/knowledge/documents` → 包含该文档且 `chunk_count` 一致。

5. **持久化**：重启 uvicorn 后 `GET /documents` 仍能列出文档，`/search` 仍可召回。

6. **删除**：`DELETE /documents/{doc_id}` → `{"deleted": true}`；再次 GET → 404；`/search` 不再返回该文档。

注意：首次入库会触发模型下载与加载，耗时较长；若 HuggingFace 不可达，先设置 `$env:HF_ENDPOINT="https://hf-mirror.com"`。
