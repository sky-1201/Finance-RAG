import json
import logging
import os
import shutil
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest
from app.services.agent import FinancialAgentService

from fastapi import File, UploadFile, BackgroundTasks
from pydantic import BaseModel
# 别忘了导入 Ingestion Service
from app.services.ingestion import DocumentIngestionService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# 实例化全局单例 Agent 服务
# 这样保证每次请求复用同一个大模型和数据库连接池
agent_service = FinancialAgentService()


@router.post("/chat/stream", summary="Agentic RAG 流式问答接口")
async def chat_stream_endpoint(request: ChatRequest):
    """
    接收用户查询，并以 SSE (Server-Sent Events) 格式流式返回 Agent 的思考与回答。
    """
    logger.info(f"🌐 API 接收到流式请求: {request.query}")

    async def event_generator():
        try:
            # 使用最新的 LangChain astream_events API 捕获底层流式输出
            # version="v2" 是 LangChain 强推荐的新版事件 API
            # 🌟 核心改动：适配 LangGraph 的消息结构
            # 🌟 核心改动：把 system_prompt 放在 user_query 前面一起传给大模型
            async for event in agent_service.agent_executor.astream_events(
                    {
                        "messages": [
                            ("system", agent_service.system_prompt),
                            ("user", request.query)
                        ]
                    },
                    version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        # 严格遵循 SSE 协议格式：data: {"chunk": "..."}\n\n
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # 结束标志
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"❌ 流式输出异常: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': '服务器内部推理错误'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class UploadResponse(BaseModel):
    message: str
    filename: str


# 修改后台任务的参数，接收原始文件名
def process_and_ingest_document(file_path: str, original_filename: str):
    logger.info(f"⏳ 后台任务开始：处理文件 {original_filename}")
    try:
        ingestion_service = DocumentIngestionService()
        # 把原始文件名也传进去
        ingestion_service.run_pipeline(pdf_path=file_path, original_filename=original_filename)
        logger.info(f"✅ 后台任务完成：文件 {original_filename} 已成功入库。")
    except Exception as e:
        logger.error(f"❌ 后台任务失败：处理 {original_filename} 时发生错误: {str(e)}", exc_info=True)


@router.post("/upload", response_model=UploadResponse, summary="上传财报 PDF 并入库")
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持上传 PDF 文件")

    logger.info(f"📥 接收到文件上传请求: {file.filename}")

    raw_dir = settings.RAW_DATA_PATH
    os.makedirs(raw_dir, exist_ok=True)

    # 🌟 核心防坑机制：用安全的 UUID 替换掉不安全的中文文件名存入硬盘
    safe_filename = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(raw_dir, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"文件保存失败: {str(e)}")
        raise HTTPException(status_code=500, detail="文件保存失败")
    finally:
        await file.close()

    # 把安全的物理路径，和用户上传的原始中文名，一起传给处理任务
    background_tasks.add_task(process_and_ingest_document, file_path, file.filename)

    return UploadResponse(
        message="文件上传成功，系统正在后台全力解析入库，请稍后进行提问。",
        filename=file.filename
    )