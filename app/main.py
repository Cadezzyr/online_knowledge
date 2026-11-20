from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # 启动/关闭时的资源初始化与清理写在这里（如数据库连接池、HTTP 客户端）
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(api_router, prefix=settings.api_v1_prefix)

# 接口测试页：浏览器打开 http://127.0.0.1:8000/demo/ ，同源访问无需配置 CORS
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
