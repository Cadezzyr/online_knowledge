import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app_es.api.router import api_router
from app_es.core.config import settings
from app_es.core.es import close_es_client, ensure_indices


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # ES 暂不可用不阻断启动；依赖是否就绪由 /knowledge/health 与业务请求反映
    try:
        ensure_indices()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Elasticsearch 初始化失败: %s", exc
        )
    yield
    close_es_client()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(api_router, prefix=settings.api_v1_prefix)

# 接口测试页：浏览器打开 http://127.0.0.1:8001/demo/ ，同源访问无需配置 CORS
app.mount(
    "/demo",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="demo",
)


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    """根路径直接跳到接口测试页。"""
    return RedirectResponse(url="/demo/")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
