import streamlit as st
import requests
import json
import time

# 配置页面基础属性

st.set_page_config(page_title="智能投研 Agent", page_icon="📈", layout="centered")

# ==========================================
# 🌟 新增：侧边栏 - 文件上传与状态显示
# ==========================================
with st.sidebar:
    st.header("📂 知识库管理")
    st.markdown("请上传真实的 A 股上市公司财报 PDF。")

    uploaded_file = st.file_uploader("上传 PDF 财报", type=["pdf"])

    if uploaded_file is not None:
        if st.button("开始解析入库"):
            with st.spinner("正在上传并触发后台处理..."):
                try:
                    # 使用 requests 发送 multipart/form-data 文件
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    # 注意端口号需与 FastAPI 一致
                    upload_res = requests.post("http://127.0.0.1:8000/api/v1/upload", files=files)

                    if upload_res.status_code == 200:
                        st.success(
                            f"✅ 上传成功！\n\n**{uploaded_file.name}** 正在后台解析中。通常需要 1-3 分钟，您可以先喝杯水，稍后再提问。")
                    else:
                        st.error(f"❌ 上传失败: {upload_res.text}")
                except Exception as e:
                    st.error(f"网络连接错误，请确保后端已启动。详情: {str(e)}")

    st.divider()
    st.info("💡 **提示:** \n建议上传类似《深信服2024年半年度报告.pdf》命名格式的文件，以便系统自动提取 Metadata。")

st.title("📈 智能金融投研 Agent")
st.markdown("基于 `Docling版面分析` + `Milvus混合检索` + `Qwen代码执行` 构建")
st.divider()

# 初始化聊天历史记录 (存放在 Streamlit Session State 中)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入框
if prompt := st.chat_input("请输入您的问题，例如：计算2024年深信服的毛利率"):
    # 1. 将用户问题上屏并保存
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 准备接收 AI 回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 3. 调用 FastAPI 后端流式接口
        try:
            # 注意：这里的 URL 端口要和 FastAPI 启动的端口一致 (默认 8000)
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/chat/stream",
                json={"query": prompt},
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            # 解析 SSE 数据流
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]  # 去掉前缀

                        if data_str == "[DONE]":
                            break

                        data_json = json.loads(data_str)
                        if "chunk" in data_json:
                            full_response += data_json["chunk"]
                            # 实时更新 UI 实现打字机效果
                            message_placeholder.markdown(full_response + "▌")

            # 最终去掉光标
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"🔌 连接后端服务失败，请检查 FastAPI 是否已启动。\n报错信息: {str(e)}")
            full_response = "请求失败。"

    # 4. 保存 AI 回答到历史记录
    if full_response != "请求失败。":
        st.session_state.messages.append({"role": "assistant", "content": full_response})