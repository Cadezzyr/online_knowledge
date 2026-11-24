# Online Knowledge API

面向「网页正文提取 → 切片入库 → 检索召回」的 FastAPI 服务，内置两套可独立运行的知识库后端：

- `app/`：本地文件持久化，llama-index `SimpleVectorStore` 向量检索 + 内存 BM25 关键词检索。
- `app_es/`：以 Elasticsearch 作为数据与存储源，支持 ES kNN 向量检索、ES BM25 关键词检索和混合检索。

两套服务共用同一份配置约定、依赖环境和接口风格，可分别启动、互不影响。

## 功能特性

- 网页正文提取：基于 trafilatura，支持直接传 URL 或 HTML，返回标题、作者、时间、站点、分类/标签、正文、段落数等结构化信息。
- 长文切片入库：按中文标点切句、按字符数切片，支持自定义 `chunk_size` / `chunk_overlap`。
- 片段级召回 + 文档级聚合：检索结果既包含命中的 chunk，也按文档聚合还原整篇原文。
- 三种（ES 后端）或两种（本地后端）检索模式：向量、关键词、混合。
- 混合检索权重可调：`alpha` 控制向量路权重，关键词路取 `1 - alpha`（相对分数融合）。
- 可选 cross-encoder 精排：`rerank=true` 时加载本地 bge-reranker 模型对候选重排序。
- 文档 CRUD：列表、详情（可选返回分片向量）、删除。
- 自带接口测试台：`/demo/` 提供网页提取、切片入库、检索、文档管理四个面板。

## 技术栈

| 类别 | 选型 |
| --- | --- |
| Web 框架 | FastAPI + Uvicorn |
| 配置 | pydantic-settings（`.env`） |
| 正文提取 | trafilatura |
| 切分 / 向量 / 融合 / 精排 | llama-index-core、llama-index-embeddings-huggingface、llama-index-retrievers-bm25 |
| 向量模型 | 默认 bge-small-zh-v1.5（512 维） |
| 精排模型 | 默认 bge-reranker-v2-m3（本地目录可配） |
| ES 后端 | Elasticsearch 8.x + 官方 elasticsearch-py（`>=8.13,<9`） |

## 目录结构

```text
online_knowledge/
├── app/                        # 后端一：本地文件持久化
│   ├── main.py                 # FastAPI 入口（/health、/demo、根路径跳转）
│   ├── core/config.py          # Settings
│   ├── api/
│   │   ├── router.py           # 路由汇总（/api/v1）
│   │   └── routes/
│   │       ├── web_text.py     # 网页正文提取
│   │       └── knowledge.py    # 知识库接口
│   ├── schemas/                # 请求/响应模型
│   ├── services/
│   │   ├── web_text.py         # trafilatura 封装
│   │   └── knowledge.py        # SimpleVectorStore + BM25 + 融合 + 精排
│   └── static/index.html       # 接口测试台
├── app_es/                     # 后端二：Elasticsearch
│   ├── main.py                 # 入口（启动时幂等建索引）
│   ├── core/
│   │   ├── config.py           # 含 ES_* 配置
│   │   └── es.py               # ES 客户端、索引映射、健康检查
│   ├── api/routes/knowledge.py # 知识库接口 + /knowledge/health
│   ├── services/
│   │   ├── knowledge.py        # ES 存取 + 三种检索
│   │   └── retrievers.py       # ES kNN / BM25 BaseRetriever
│   ├── schemas/                # 与 app 对齐，mode 增加 keyword
│   └── static/index.html       # ES 版测试台
├── data/                       # 本地模型与索引数据（git 忽略）
│   ├── models/                 # embedding / rerank 模型
│   └── knowledge/              # app 的索引持久化目录
├── sources/                    # 本地 editable 依赖源码（git 忽略）
├── .trae/documents/            # 接口设计说明
├── .env.example
├── pyproject.toml
└── uv.lock
```

## 两套后端对比

| 能力 | `app/`（本地） | `app_es/`（Elasticsearch） |
| --- | --- | --- |
| 存储 | 本地文件（`data/knowledge`） | ES 索引 `knowledge_documents` + `knowledge_chunks` |
| 向量检索 | `SimpleVectorStore` | ES kNN（`dense_vector`，cosine） |
| 关键词检索 | 内存 BM25（按语料重建） | ES BM25（`multi_match`） |
| 混合检索 | 向量 + BM25，`QueryFusionRetriever` 相对分数融合 | 同左，两路都来自 ES |
| 检索模式 | `vector` / `hybrid` | `vector` / `keyword` / `hybrid` |
| 文档过滤 | 召回后按 `doc_ids` 过滤 | ES 查询层 `terms` 过滤 |
| 健康检查 | 无 | `GET /api/v1/knowledge/health` |
| 数据持久化 | 进程重启后仍在 | ES 集群持久化 |
| 启动依赖 | 本地模型目录 | 本地模型 + 可用 ES 实例 |

> 两者接口路径一致，但 `app` 的检索请求不接受 `mode=keyword`。

## 快速开始

### 1. 环境要求

- Python >= 3.11
- （`app_es` 需要）Elasticsearch 8.13+，本地或远端均可
- 首次运行需准备 embedding / rerank 模型（本地目录或可访问 HuggingFace）

### 2. 安装依赖

```powershell
# 推荐：使用 uv（会按 uv.lock 安装）
uv sync

# 或者在已有虚拟环境中安装
pip install -e .
```

`pyproject.toml` 中 `[tool.uv.sources]` 把 `trafilatura`、`llama-index-core`、`llama-index-retrievers-bm25` 指向了本地 `sources/` 目录（该目录被 `.gitignore` 忽略）。如果拿到的是不含 `sources/` 的副本，需要改用 PyPI 版本或补齐对应源码。

### 3. 配置

复制 `.env.example` 为 `.env` 后按需修改。常用项：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_NAME` | `Online Knowledge API` | 服务标题 |
| `API_V1_PREFIX` | `/api/v1` | 路由前缀 |
| `DEBUG` | `false` | FastAPI 调试开关 |
| `KNOWLEDGE_STORE_DIR` | `data/knowledge` | 本地后端索引目录 |
| `KNOWLEDGE_CHUNK_SIZE` | `512` | 默认切片长度（字符） |
| `KNOWLEDGE_CHUNK_OVERLAP` | `64` | 默认切片重叠（字符） |
| `KNOWLEDGE_TOP_K` | `5` | 默认召回条数 |
| `KNOWLEDGE_LIST_LIMIT` | `500` | ES 后端文档列表上限 |
| `EMBEDDING_MODEL` | `bge-small-zh-v1.5` | HuggingFace 模型名或本地目录 |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` |
| `EMBEDDING_BATCH_SIZE` | `8` | 向量化批大小 |
| `RERANK_MODEL` | 本地 `bge-reranker-v2-m3` | 精排模型目录或模型名 |
| `RERANK_TOP_N` | `50` | 精排保留候选数 |
| `ES_URL` | `http://localhost:9200` | ES 地址 |
| `ES_USERNAME` / `ES_PASSWORD` | 空 | Basic Auth（可选） |
| `ES_API_KEY` | 空 | API Key（优先于用户名密码） |
| `ES_VERIFY_CERTS` / `ES_CA_CERTS` | `true` / 空 | HTTPS 证书校验 |
| `ES_REQUEST_TIMEOUT` | `10` | 请求超时（秒） |
| `ES_INDEX_DOCUMENTS` | `knowledge_documents` | 文档索引名 |
| `ES_INDEX_CHUNKS` | `knowledge_chunks` | 分片索引名 |
| `ES_ANALYZER` | `standard` | 文本分词器；中文可改 `ik_max_word`（需 IK 插件） |
| `ES_NUMBER_OF_SHARDS` | `1` | 主分片数 |
| `ES_NUMBER_OF_REPLICAS` | `0` | 副本数 |
| `ES_EMBEDDING_DIM` | `512` | 必须与 embedding 模型维度一致 |

模型提示：`EMBEDDING_MODEL` 可以写本地目录（如 `data/models/bge-small-zh-v1.5`）；网络受限时可设置 `HF_ENDPOINT=https://hf-mirror.com` 后再下载。

### 4. 启动

```powershell
# 本地后端（端口 8000）
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Elasticsearch 后端（端口 8001）
.\.venv\Scripts\python.exe -m uvicorn app_es.main:app --host 127.0.0.1 --port 8001
```

- Swagger：`/docs`
- 测试台：`/demo/`
- 健康检查：`/health`

`app_es` 启动时会尝试幂等创建两个索引；ES 暂不可用时只打印告警，不会阻断启动。索引会在首次业务请求时再次检查，ES 恢复后无需重启。

## 接口说明

统一前缀 `/api/v1`（可通过 `API_V1_PREFIX` 修改）。

### 网页正文提取

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/web_text/url_text` | 下载 URL 并提取正文；下载失败 502，无可提取正文 422 |
| POST | `/web_text/html_text` | 直接提取传入 HTML 的正文；无可提取正文 422 |

请求示例：

```json
{
  "url": "https://example.com/article",
  "options": {
    "include_links": true,
    "favor_precision": true,
    "target_language": "zh"
  }
}
```

`options` 支持 trafilatura 的 `include_comments`、`include_tables`、`include_images`、`include_links`、`favor_precision`、`favor_recall`、`deduplicate`、`target_language`，未传时使用 trafilatura 默认值。响应包含 `url`、`title`、`author`、`date`、`sitename`、`description`、`categories`、`tags`、`language`、`text`、`comments`、`char_count`、`paragraph_count`。

### 知识库

| 方法 | 路径 | 说明 | 主要错误 |
| --- | --- | --- | --- |
| POST | `/knowledge/documents` | 长文切片入库 | 409 已存在 |
| POST | `/knowledge/search` | 片段召回并聚合 | 503 精排不可用 / ES 不可用 |
| GET | `/knowledge/documents` | 文档列表 | — |
| GET | `/knowledge/documents/{doc_id}` | 文档详情，`include_vectors=true` 返回向量 | 404 |
| DELETE | `/knowledge/documents/{doc_id}` | 删除文档及分片 | 404 |
| GET | `/knowledge/health` | ES 连接与索引状态（仅 `app_es`） | — |

入库示例：

```json
{
  "text": "需要入库的长文本……",
  "title": "示例文档",
  "doc_id": "demo-001",
  "chunk_size": 256,
  "chunk_overlap": 32
}
```

`doc_id` 不传会自动生成 uuid4。响应返回 `doc_id`、`chunk_count` 与每个分片的 `chunk_id` / `chunk_index` / `text`。

检索示例：

```json
{
  "query": "向量检索与关键词检索的区别",
  "top_k": 5,
  "doc_ids": ["demo-001"],
  "mode": "hybrid",
  "alpha": 0.7,
  "rerank": false,
  "include_text": true
}
```

- `mode`：`app_es` 支持 `vector` / `keyword` / `hybrid`；`app` 支持 `vector` / `hybrid`。
- `alpha`：仅 `hybrid` 生效，向量路权重，关键词路为 `1-alpha`。
- `doc_ids`：限定检索范围（ES 后端在查询层过滤）。
- 响应包含 `hits`（片段级）与 `documents`（按文档聚合，含 `full_text`、`best_score`、`matched_chunks`）。

健康检查示例（`app_es`）：

```json
{
  "status": "unavailable",
  "url": "http://localhost:9200",
  "cluster_name": null,
  "version": null,
  "documents_index": false,
  "chunks_index": false
}
```

接口正常时 `status` 为 `ok`，并返回集群名、版本与两个索引是否存在。

## 检索机制

1. **切片**：`SentenceSplitter` 使用自定义字符 tokenizer 和中文标点正则切句，避免首次调用联网下载 NLTK 数据；切片长度按字符计算。
2. **向量化**：`HuggingFaceEmbedding` 批量编码分片文本。
3. **召回**：
   - 本地后端：`VectorStoreIndex.as_retriever` 做向量召回；`hybrid` 时额外构建内存 BM25（中文按单字 tokenize），用 `QueryFusionRetriever` 相对分数融合。
   - ES 后端：`ESVectorRetriever` 走 kNN（`num_candidates = max(k*10, 100)`）；`ESKeywordRetriever` 走 `multi_match`（`content` 与 `title^2`）；`hybrid` 同样用相对分数融合。
4. **候选池**：先取 `max(top_k*4, 20)` 条，给精排留素材。
5. **精排**：`rerank=true` 时用 `SentenceTransformerRerank` 对候选重排，最后截断到 `top_k`。
6. **聚合**：按 `doc_id` 分组，本地后端从 docstore、ES 后端从 `knowledge_documents` 还原原文与分片统计。

## Elasticsearch 索引设计

`knowledge_chunks`（检索单元）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_id` / `doc_id` / `ref_doc_id` | keyword | 分片与文档标识 |
| `chunk_index` | integer | 分片顺序 |
| `title` | text + keyword 子字段 | 标题，关键词检索加权 |
| `content` | text（`ES_ANALYZER`） | 分片正文 |
| `embedding` | dense_vector（`ES_EMBEDDING_DIM`，cosine） | 向量 |
| `created_at` | date | 写入时间 |

`knowledge_documents`（原文与统计）：`doc_id`、`title`、`full_text`、`char_count`、`chunk_count`、`created_at`。

写入使用 `elasticsearch.helpers.bulk` 并带 `refresh="wait_for"`，接口返回后即可检索。

## 测试台

启动任一服务后打开 `http://127.0.0.1:<port>/demo/`：

1. **网页正文提取**：输入 URL 或粘贴 HTML，查看结构化元数据与正文。
2. **切片入库**：把提取结果填入或直接粘贴文本，设置 `doc_id` / 标题 / 切片参数。
3. **片段检索**：设置查询、`top_k`、模式、`alpha`、是否精排、限定 `doc_ids`。
4. **文档管理**：列出文档、查看分片与向量、删除文档。

`app` 页面的检索模式为 `vector` / `hybrid`；`app_es` 页面额外提供 `keyword`。

## 常见问题

**ES 不可用会怎样？**
`app_es` 启动只打印告警；`/api/v1/knowledge/health` 返回 `status=unavailable`，业务接口返回 503 并附带连接错误信息。检查 `ES_URL`、网络、账号或证书配置即可，ES 恢复后无需重启服务。

**向量维度不一致？**
`ES_EMBEDDING_DIM` 必须等于 embedding 模型输出维度（bge-small-zh-v1.5 为 512）。不一致时入库会直接报错。已存在的索引不会因环境变量变化而改映射，需要重建索引。

**中文关键词召回效果一般？**
默认 `standard` 分词对中文按字切分。可安装 IK 分词插件、将 `ES_ANALYZER` 设为 `ik_max_word`，并用新索引名或删除旧索引后重建；索引映射只在创建时生效。

**精排返回 503？**
`RERANK_MODEL` 指向的目录不存在或加载失败。修正路径或设为可下载的 HuggingFace 模型名；也可以先 `rerank=false`。

**首次启动较慢？**
依赖导入与本地 embedding/rerank 模型加载需要时间，属正常现象。

## 已知限制

- 未内置鉴权与多租户隔离，生产部署需自行加网关或中间件。
- 本地后端为单进程文件存储，多实例并发写不适用；ES 后端依赖集群能力。
- 同一 `doc_id` 不支持覆盖更新，需先删除再入库。
- `GET /documents/{doc_id}`（ES 后端）最多返回 10000 个分片，文档列表受 `KNOWLEDGE_LIST_LIMIT` 限制。
- 未包含前缀检索、稀疏向量、ES 原生 RRF/加权 hybrid 等扩展模式。
