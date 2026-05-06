from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """前端发给后端的聊天请求模型"""
    query: str = Field(..., description="用户的提问内容", min_length=1, max_length=1000)

class ChatResponse(BaseModel):
    """标准同步响应模型（备用）"""
    answer: str