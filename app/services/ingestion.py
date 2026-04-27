import random
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

# 🌟 终极防掉线方案：纯原生 PyMilvus
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

        # 3. 初始化切分器
        self.parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
            strip_headers=False
        )

        # 🌟 核心防丢失机制 1：父块物理兜底切分器
        # 确保哪怕遇到十几万字不分段的变态财报表格，父块也绝对不会超过 Milvus 65535 字符的物理极限，彻底杜绝数据截断丢失！
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
        """完整的端到端入库流程"""
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

            # 🌟 新增：对切出来的超大父块进行物理兜底，确保父块信息 100% 留存
            safe_parent_docs = []
            for doc in parent_docs:
                if len(doc.page_content) > 40000:
                    safe_parent_docs.extend(self.parent_fallback_splitter.split_documents([doc]))
                else:
                    safe_parent_docs.append(doc)

            file_meta = self._extract_metadata(display_name)

            final_docs = []
            # 注意：这里改成了遍历安全的 safe_parent_docs
            for p_doc in safe_parent_docs:
                parent_id = str(uuid.uuid4())
                p_doc.metadata.update(file_meta)
                p_doc.metadata["parent_id"] = parent_id
                p_doc.metadata["doc_level"] = "parent"

                final_docs.append(p_doc)

                child_chunks = self.child_splitter.split_documents([p_doc])
                for c_doc in child_chunks:
                    c_doc.metadata["doc_level"] = "child"
                    final_docs.append(c_doc)

            # ==========================================
            # C. 存入 Milvus (引入“非对称假向量”架构优化)
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

            # 2. 检查并创建原生的表结构
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

            # 3. 🌟 核心架构优化：非对称 Embedding
            # 将父块和子块分离，只让子块去消耗 API 生成真实向量
            child_docs = [doc for doc in final_docs if doc.metadata["doc_level"] == "child"]
            parent_docs = [doc for doc in final_docs if doc.metadata["doc_level"] == "parent"]

            logger.info(
                f"🧠 正在调用模型生成子块向量 (共 {len(child_docs)} 个)，父块 (共 {len(parent_docs)} 个) 直接跳过...")

            # --- a. 对子块进行真实的 Embedding（分批请求 API） ---
            child_texts = [doc.page_content for doc in child_docs]
            child_embeddings = []
            batch_size = 10

            for i in range(0, len(child_texts), batch_size):
                batch_texts = child_texts[i:i + batch_size]
                logger.info(
                    f"⏳ 正在向 API 请求子块向量: 进度 {i + 1} ~ {min(i + batch_size, len(child_texts))} / {len(child_texts)}")

                # 阿里 API 单次文本最大长度为 8192
                batch_texts_for_embed = [t[:8000] for t in batch_texts]
                batch_embeddings = self.embeddings.embed_documents(batch_texts_for_embed)
                child_embeddings.extend(batch_embeddings)

            # --- b. 对父块生成“假向量”（瞬间完成，零计算成本） ---
            # ❌ 删掉这行引发黑洞的纯相同向量：
            # dummy_vector = [0.1] * 1024
            # parent_embeddings = [dummy_vector for _ in parent_docs]

            # ✅ 替换为：生成带有随机噪音的假向量，强制打散它们在 HNSW 图中的位置！
            parent_embeddings = [
                [random.uniform(-0.1, 0.1) for _ in range(1024)]
                for _ in parent_docs
            ]

            parent_texts = [doc.page_content for doc in parent_docs]

            # 4. 组装并原生插入数据
            insert_data = [
                child_texts + parent_texts,  # 文本列
                child_embeddings + parent_embeddings,  # 向量列 (前段真实，后段作假)
                [doc.metadata for doc in child_docs] + [doc.metadata for doc in parent_docs]  # 元数据列
            ]

            collection.insert(insert_data)
            collection.flush()

            logger.info("✅ 极其纯净的 Native Ingestion Pipeline 执行成功，成本节省一半！")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)