import logging
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- 基础配置 ---
    APP_NAME: str = "Financial_Agentic_RAG"
    DEBUG: bool = True

    # --- 模型 API 配置 ---
    DASHSCOPE_API_KEY: str  # 必须在 .env 中配置，否则启动报错
    EMBEDDING_MODEL: str = "text-embedding-v4"  # 阿里最新的大模型嵌入 API
    LLM_MODEL: str = "qwen-max"

    # --- Milvus 向量库配置 ---
    MILVUS_URI: str = "./data/milvus_financial.db"  # 本地测试使用轻量级 Lite
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: str = "19531"
    COLLECTION_NAME: str = "financial_reports_parent_child"

    # --- Chunking 策略配置 (即将用到) ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # --- 路径配置 ---
    RAW_DATA_PATH: str = "data/raw"
    PROCESSED_DATA_PATH: str = "data/processed"

    # Pydantic V2 读取环境变量的标准写法
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# 实例化单例，整个项目都可以 import 这个 settings
settings = Settings()

# 全局日志配置 (放到 config 里统一管理)
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)