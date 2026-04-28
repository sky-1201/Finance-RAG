import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

# 🌟 加载 .env 文件
load_dotenv()

# 从环境变量中动态读取密码和地址
PG_USER = os.getenv("PG_USER", "rag_user")
PG_PASSWORD = os.getenv("PG_PASSWORD", "rag_password")
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "rag_db")

# 动态拼接数据库连接字符串
SQLALCHEMY_DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# 创建引擎
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 定义父块的数据库表
class ParentDocument(Base):
    __tablename__ = "parent_documents"
    id = Column(String, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    meta_data = Column("metadata", JSONB)

def init_db():
    print(f"⏳ 正在连接 PostgreSQL ({PG_HOST}:{PG_PORT}) 并初始化表结构...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ 数据库表 `parent_documents` 准备就绪！")
    except Exception as e:
        print(f"❌ 连接失败，请检查 Docker 是否启动以及账号密码是否正确。报错信息: {e}")

if __name__ == "__main__":
    init_db()