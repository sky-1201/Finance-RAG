import logging
# 🌟 1. 删掉旧的、有 Bug 的 ChatTongyi
# from langchain_community.chat_models import ChatTongyi
# 🌟 2. 引入极度稳定的 ChatOpenAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from app.core.config import settings
from app.tools.finance_repl import python_repl_tool
from app.tools.retriever_tool import financial_retriever_tool

logger = logging.getLogger(__name__)


class FinancialAgentService:
    def __init__(self):
        # 🌟 3. 换成 OpenAI 兼容模式对接阿里通义大模型！
        # 这个底层解析器完美支持流式输出和工具调用，绝不死锁。
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,  # 依然是你的 qwen3-max
            api_key=settings.DASHSCOPE_API_KEY,
            # 关键：把请求发给阿里的兼容服务器，而不是美国 OpenAI
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.01,
            streaming=True
        )
        self.tools = [financial_retriever_tool, python_repl_tool]

        self.system_prompt = (
            "你是一个顶级的金融投研 AI 分析师。你的回答必须基于客观事实，严谨且专业。\n"
            "你拥有以下工具来辅助你：\n"
            "1. `financial_retriever_tool`: 用于从财报库中检索客观数据和文本上下文。\n"
            "2. `python_repl_tool`: 用于执行 Python 代码进行精确计算。\n\n"
            "【严格遵守的工作流】\n"
            "步骤一：如果用户询问某项财务指标，先用 `financial_retriever_tool` 查出具体数值。\n"
            "步骤二：如果问题涉及数学计算（例如：毛利率、同比增加百分比），绝对不要自己心算！必须写出 Python 代码，并通过 `python_repl_tool` 计算。\n"
            "步骤三：综合检索到的事实和计算出的结果，生成结构化回答。\n\n"
            "警告1：不要编造任何财务数据！\n"
            "警告2：在调用工具时，如果用户没有明确指明具体年份或公司名称，对应的工具参数必须留空，绝不允许自行脑补或使用示例值！"
        )

        self.agent_executor = create_react_agent(self.llm, self.tools)

    def chat(self, query: str) -> str:
        """同步测试接口"""
        logger.info("=" * 50)
        logger.info(f"👤 用户提问: {query}")
        try:
            messages = [
                ("system", self.system_prompt),
                ("user", query)
            ]
            response = self.agent_executor.invoke({"messages": messages})
            return response["messages"][-1].content
        except Exception as e:
            logger.error(f"❌ Agent 崩溃: {str(e)}", exc_info=True)
            return "分析系统遇到内部错误，请稍后重试。"