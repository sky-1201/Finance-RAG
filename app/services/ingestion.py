import logging
from pathlib import Path
from typing import Optional, Tuple
import re
import uuid

# LangChain 相关
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

# Milvus 原生包
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# 自定义组件
from app.core.config import settings
from docling.document_converter import DocumentConverter

# 🌟 新增：引入 Postgres 数据库连接和表模型
from app.database import SessionLocal, ParentDocument

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    def __init__(self):
        # 1. 初始化 PDF 解析器
        self.converter = DocumentConverter()

        # 2. 初始化 Embedding 模型 (阿里通义千问)
        self.embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )

        # 3. 初始化切分器
        self.parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
            strip_headers=False
        )

        # 虽然用双库了，但保留一个物理兜底（比如按 40000 切），确保极个别变态文本也能安全落盘
        self.parent_fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000,
            chunk_overlap=1000
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def _extract_metadata(self, file_name: str) -> dict:
        year_match = re.search(r'(20\d{2})', file_name)
        company_match = re.search(r'^(.*?)(?:20\d{2})', file_name)
        return {
            "year": year_match.group(1) if year_match else "未知",
            "company": company_match.group(1) if company_match else "未知",
            "source": file_name
        }

    def run_pipeline(self, pdf_path: str, original_filename: str = None, page_range: Optional[Tuple[int, int]] = None):
        """完整的端到端入库流程 (双库架构)"""
        try:
            path = Path(pdf_path)
            display_name = original_filename if original_filename else path.name

            # ==========================================
            # A. 解析 PDF (Docling)
            # ==========================================
            logger.info(f"Step 1: Parsing PDF {display_name}...")

            if page_range is None:
                doc_result = self.converter.convert(path)
            else:
                doc_result = self.converter.convert(path, page_range=page_range)

            md_text = doc_result.document.export_to_markdown()

            # ==========================================
            # B. 父子块切分与 Metadata 组装
            # ==========================================
            logger.info("Step 2: Parent and Child splitting...")
            parent_docs = self.parent_splitter.split_text(md_text)

            # 物理兜底，防止出现极端巨大的单一块
            safe_parent_docs = []
            for doc in parent_docs:
                if len(doc.page_content) > 40000:
                    safe_parent_docs.extend(self.parent_fallback_splitter.split_documents([doc]))
                else:
                    safe_parent_docs.append(doc)

            file_meta = self._extract_metadata(display_name)

            child_docs = []

            # 为父块分配 parent_id，并切出子块
            for p_doc in safe_parent_docs:
                parent_id = str(uuid.uuid4())
                p_doc.metadata.update(file_meta)
                p_doc.metadata["parent_id"] = parent_id
                p_doc.metadata["doc_level"] = "parent"

                # 切分子块
                child_chunks = self.child_splitter.split_documents([p_doc])
                for c_doc in child_chunks:
                    c_doc.metadata.update(file_meta)  # 确保子块也有年份等基础信息
                    c_doc.metadata["parent_id"] = parent_id
                    c_doc.metadata["doc_level"] = "child"
                    child_docs.append(c_doc)

            # ==========================================
            # C. 双库落盘：父块 -> PostgreSQL | 子块 -> Milvus
            # ==========================================
            logger.info("Step 3: Executing Compute & Storage Decoupling Pipeline...")

            # ----------------------------------------------------
            # 🟢 分支 1：将超级父块存入 PostgreSQL 存储层
            # ----------------------------------------------------
            logger.info("📦 开始将完整父块写入 PostgreSQL 存储层...")
            db = SessionLocal()
            try:
                postgres_records = []
                for p_doc in safe_parent_docs:
                    record = ParentDocument(
                        id=p_doc.metadata["parent_id"],
                        content=p_doc.page_content,
                        meta_data=p_doc.metadata
                    )
                    postgres_records.append(record)

                db.add_all(postgres_records)
                db.commit()
                logger.info(f"✅ 成功将 {len(postgres_records)} 个超级父块安全落盘至 PostgreSQL！")
            except Exception as e:
                db.rollback()
                logger.error(f"❌ PostgreSQL 写入失败: {str(e)}")
                raise e
            finally:
                db.close()

            # ----------------------------------------------------
            # 🔵 分支 2：将子块及其向量存入 Milvus 计算层
            # ----------------------------------------------------
            logger.info(f"🧠 开始处理子块及其向量...")

            try:
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT
                )
            except Exception as e:
                logger.warning(f"⚠️ PyMilvus 连接复用提示: {e}")

            collection_name = settings.COLLECTION_NAME

            if not utility.has_collection(collection_name):
                logger.info(f"📦 Collection '{collection_name}' 不存在，正在创建表结构...")
                fields = [
                    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]
                schema = CollectionSchema(fields, "Financial RAG Document Store", enable_dynamic_field=True)
                collection = Collection(collection_name, schema)

                index_params = {
                    "index_type": "AUTOINDEX",
                    "metric_type": "L2",
                    "params": {}
                }
                collection.create_index("vector", index_params)
                logger.info("📦 表和向量索引创建完成！")
            else:
                collection = Collection(collection_name)

            # --- 对子块进行真实的 Embedding ---
            logger.info(f"⏳ 正在向 API 请求 {len(child_docs)} 个子块向量...")
            child_texts = [doc.page_content for doc in child_docs]
            child_embeddings = []
            batch_size = 10

            for i in range(0, len(child_texts), batch_size):
                batch_texts = child_texts[i:i + batch_size]
                logger.info(f"   进度: {i + 1} ~ {min(i + batch_size, len(child_texts))} / {len(child_texts)}")
                batch_texts_for_embed = [t[:8000] for t in batch_texts]
                batch_embeddings = self.embeddings.embed_documents(batch_texts_for_embed)
                child_embeddings.extend(batch_embeddings)

            # 组装并原生插入 Milvus（注意：现在只有子块了，不再需要“假向量”逻辑）
            milvus_insert_data = [
                child_texts,  # 文本列
                child_embeddings,  # 向量列
                [doc.metadata for doc in child_docs]  # 元数据列 (包含 parent_id)
            ]

            collection.insert(milvus_insert_data)
            collection.flush()

            logger.info("✅ 完美的双库解耦入库完成！计算(Milvus)与存储(Postgres)彻底分离。")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)