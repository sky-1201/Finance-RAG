import app.core.logger  # 第一行导入，确保全局日志初始化立即生效
import logging
from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import router

logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    description="基于版面分析与 Agentic 架构的智能金融投研 RAG 系统",
    version="1.0.0"
)

# 挂载业务路由
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 FastAPI 后端服务已启动！")
    logger.info(f"📚 接口文档地址: http://{settings.API_HOST}:{settings.API_PORT}/docs")