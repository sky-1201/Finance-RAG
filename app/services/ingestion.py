import re
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from app.core.config import settings

# 🌟 终极防掉线方案：直接引入 PyMilvus 底层包
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

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

        # 3. 初始化两种切分器
        self.parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
            strip_headers=False
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def _extract_metadata(self, file_name: str) -> dict:
        """从文件名中提取年份和公司名"""
        year_match = re.search(r'(20\d{2})', file_name)
        company_match = re.search(r'^(.*?)(?:20\d{2})', file_name)
        return {
            "year": year_match.group(1) if year_match else "未知",
            "company": company_match.group(1) if company_match else "未知",
            "source": file_name
        }

    def run_pipeline(self, pdf_path: str, original_filename: str = None, page_range: Optional[Tuple[int, int]] = None):
        """完整的端到端入库流程"""
        try:
            path = Path(pdf_path)
            # 如果没传原始名，就用路径名兜底
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

            # 🌟 核心修复：用原始的中文文件名去提取年份和公司！
            file_meta = self._extract_metadata(display_name)

            # ... 保持后面的代码完全不变 ...

            final_docs = []
            for p_doc in parent_docs:
                # 为父块生成唯一 ID
                parent_id = str(uuid.uuid4())
                p_doc.metadata.update(file_meta)
                p_doc.metadata["parent_id"] = parent_id
                p_doc.metadata["doc_level"] = "parent"

                # 🌟 重要：把父块本身也加入入库列表，供后续重排召回
                final_docs.append(p_doc)

                # 将父块切分为子块
                child_chunks = self.child_splitter.split_documents([p_doc])
                for c_doc in child_chunks:
                    c_doc.metadata["doc_level"] = "child"
                    final_docs.append(c_doc)

            # ==========================================
            # C. 存入 Milvus (完全使用原生 PyMilvus 绕过 Bug)
            # ==========================================
            logger.info(f"Step 3: Upserting {len(final_docs)} chunks to Milvus natively...")

            # 1. 强行建立原生连接
            try:
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT
                )
            except Exception as e:
                logger.warning(f"⚠️ PyMilvus 连接复用提示: {e}")

            collection_name = settings.COLLECTION_NAME

            # 2. 检查并创建原生的表结构 (Schema)
            if not utility.has_collection(collection_name):
                logger.info(f"📦 Collection '{collection_name}' 不存在，正在创建表结构...")
                fields = [
                    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    # 注意：DashScope 的 text-embedding-v3 模型输出维度是 1024
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                    # 开启动态 Schema 存储复杂的 Metadata 字典
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]
                schema = CollectionSchema(fields, "Financial RAG Document Store", enable_dynamic_field=True)
                collection = Collection(collection_name, schema)

                # 建向量索引加速
                index_params = {
                    "index_type": "AUTOINDEX",
                    "metric_type": "L2",
                    "params": {}
                }
                collection.create_index("vector", index_params)
                logger.info("📦 表和向量索引创建完成！")
            else:
                collection = Collection(collection_name)

            # 3. 准备数据进行 Embedding
            logger.info("🧠 正在调用模型生成向量...")
            texts = [doc.page_content for doc in final_docs]
            metadatas = [doc.metadata for doc in final_docs]

            # 批量获取 Embedding
            embeddings = self.embeddings.embed_documents(texts)

            # 4. 组装并原生插入数据
            insert_data = [
                texts,  # text 列
                embeddings,  # vector 列
                metadatas  # metadata 列
            ]
            collection.insert(insert_data)
            collection.flush()  # 强制数据落盘

            logger.info("✅ Native Ingestion Pipeline Finished Successfully!")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)