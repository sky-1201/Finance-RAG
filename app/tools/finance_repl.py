import sys
import io
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# 使用 LangChain 的 @tool 装饰器，将普通 Python 函数包装成大模型认识的工具
@tool
def python_repl_tool(code: str) -> str:
    """
    一个 Python 解释器工具。当你需要进行任何财务数据的数学计算（如加减乘除、毛利率、同比增长等）时，必须使用此工具。
    输入必须是合法的 Python 代码。
    注意：为了让我看到执行结果，你必须在代码的最后使用 print() 将结果打印出来。
    """
    logger.info("=" * 40)
    logger.info(f"🤖 触发 Agent 工具: 正在执行大模型生成的 Python 代码 👇\n{code}")
    logger.info("=" * 40)

    # 核心黑科技：将系统的标准输出 (控制台) 劫持到内存里，
    # 这样才能把大模型 print() 出来的内容抓取下来，并作为字符串返回给它
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()

    try:
        # 执行代码
        # 面试加分点注意：真实的企业生产环境中，exec 极度危险（防止大模型写出删库跑路的代码）。
        # 这里作为本地 MVP 演示，我们直接执行。如果是上云，这步一定要放在 Docker 沙箱或 WASM 环境中。
        exec(code, {"__builtins__": __builtins__}, {})

        output = redirected_output.getvalue().strip()
        logger.info(f"✅ 工具执行成功，返回结果: {output}")

        return output if output else "代码执行成功，但没有使用 print() 输出结果。请修改代码并重试。"

    except Exception as e:
        error_msg = f"❌ 代码执行出错: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        # 把错误信息直接 return 给大模型，聪明的大模型会根据报错自己 Debug 并重新调用工具！
        return error_msg

    finally:
        # 无论成功失败，必须把标准输出还给系统，否则后续所有的日志全看不到了
        sys.stdout = old_stdout