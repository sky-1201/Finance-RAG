import sys
import io
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 🌟 拿掉 async，恢复为普通的 def
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

    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()

    try:
        exec(code, {"__builtins__": __builtins__}, {})
        output = redirected_output.getvalue().strip()
        logger.info(f"✅ 工具执行成功，返回结果: {output}")
        return output if output else "代码执行成功，但没有使用 print() 输出结果。请修改代码并重试。"

    except Exception as e:
        error_msg = f"❌ 代码执行出错: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return error_msg

    finally:
        sys.stdout = old_stdout