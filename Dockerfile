# 1. 选用官方轻量级 Python 3.10 镜像作为基础底座
FROM python:3.10-slim

# 2. 在容器内部创建一个叫 /app 的工作目录
WORKDIR /app

# 3. 设置环境变量：防止生成 .pyc 文件，并让日志直接输出到终端（不进缓冲区）
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. 安装基础系统编译工具 (有些 Python 库底层是 C 写的，需要这个才能安装成功)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. 【极其重要的一步】先拷贝依赖清单并安装
# 这样做是为了利用 Docker 的层缓存机制！只要 requirements 不变，以后改代码秒级打包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 把你本地的所有代码文件拷贝进容器的 /app 目录里
COPY . .

# 7. 暴露 FastAPI 的 8000 端口和 Streamlit 的 8501 端口
EXPOSE 8000 8501

# 8. 默认启动命令：拉起 FastAPI 后端服务器
CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]