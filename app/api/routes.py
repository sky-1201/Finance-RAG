#启动命令 uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
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
from app.services.ingestion import DocumentIngestionService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# 实例化全局单例 Agent 服务
agent_service = FinancialAgentService()


@router.post("/chat/stream", summary="Agentic RAG 流式问答接口")
async def chat_stream_endpoint(request: ChatRequest):
    logger.info(f"🌐 API 接收到流式请求: {request.query}")

    async def event_generator():
        try:
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
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

                # 🌟 捕获工具调用，给前端发送"思考中"的状态
                elif event["event"] == "on_tool_start":
                    tool_name = event["name"]
                    if tool_name == "financial_retriever_tool":
                        msg = "\n\n> 🔍 **[Agent 思考中]** 正在查阅 Milvus 知识库...\n\n"
                        yield f"data: {json.dumps({'chunk': msg})}\n\n"
                    elif tool_name == "python_repl_tool":
                        msg = "\n\n> 💻 **[Agent 思考中]** 正在编写 Python 代码进行精确计算...\n\n"
                        yield f"data: {json.dumps({'chunk': msg})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"❌ 流式输出异常: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': '服务器内部推理错误'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class UploadResponse(BaseModel):
    message: str
    filename: str


def process_and_ingest_document(file_path: str, original_filename: str):
    logger.info(f"⏳ 后台任务开始：处理文件 {original_filename}")
    try:
        ingestion_service = DocumentIngestionService()
        # 🌟 此处的 page_range=(1, 5) 用于本地防炸内存测试。若要解析全本，将其改为 None 或删掉该参数即可。
        ingestion_service.run_pipeline(
            pdf_path=file_path,
            original_filename=original_filename,
            page_range=None
        )
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

    # 🌟 用安全的 UUID 替换掉不安全的中文文件名存入硬盘
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

    # 把物理路径和原始中文名一起传给处理任务
    background_tasks.add_task(process_and_ingest_document, file_path, file.filename)

    return UploadResponse(
        message="文件上传成功，系统正在后台全力解析入库，请稍后进行提问。",
        filename=file.filename
    )