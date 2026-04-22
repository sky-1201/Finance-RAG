import logging
from langchain_core.tools import tool
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# 1. 全局变量初始设为 None，不要在这里直接实例化！
_retriever_service = None


@tool
def financial_retriever_tool(query: str, company: str = None, year: str = None) -> str:
    """
    当你需要查询真实客观的上市公司的财务数据（如营收、利润）、业务情况、风险提示等信息时，必须优先调用此工具。
    输入参数：
    - query: 具体的查询问题，例如"研发费用是多少？"、"主要业务有哪些？"
    - company: 公司名称，例如"深信服"（如果有的话，可选）
    - year: 年份，例如"2023"或"2024"（如果有的话，可选）

    返回：
    包含上下文片段的纯文本，请仔细阅读返回的文本以提取事实。
    """
    # 2. 延迟加载 (Lazy Load)：只有大模型真正调用这个工具时，才去连接数据库
    global _retriever_service
    if _retriever_service is None:
        _retriever_service = RetrievalService()

    logger.info(f"🛠️ Agent 决定调用检索工具 | 搜索词: {query} | 公司: {company} | 年份: {year}")

    try:
        # 调用检索 Pipeline
        docs = _retriever_service.run_pipeline(query=query, company=company, year=year)

        if not docs:
            return "数据库中未检索到相关财报信息。请告知用户没有查到，不要自行编造。"

        context_parts = []
        for i, d in enumerate(docs):
            source = d.metadata.get("source", "未知文件")
            score = d.metadata.get("rerank_score", "N/A")
            context_parts.append(f"--- 证据 {i + 1} [来源: {source}, 相关度: {score}] ---\n{d.page_content}\n")

        return "\n".join(context_parts)

    except Exception as e:
        logger.error(f"❌ 检索工具执行失败: {str(e)}", exc_info=True)
        return "检索系统发生内部错误，请稍后重试。"