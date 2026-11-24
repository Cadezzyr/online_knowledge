from fastapi import APIRouter
from app_es.api.routes import knowledge, web_text

api_router = APIRouter()
api_router.include_router(web_text.router)
api_router.include_router(knowledge.router)

# 后续业务接口在此注册，例如：
# from app_es.api.routes import extract
# api_router.include_router(extract.router, prefix="/extract", tags=["extract"])
