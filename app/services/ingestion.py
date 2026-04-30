import logging
from pathlib import Path
from typing import Optional, Tuple
import re
import uuid
import hashlib

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
from app.database import SessionLocal, ParentDocument,UploadedFile
from pypdf import PdfReader  # 如果没有安装，请在终端运行 pip install pypdf

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

    def _calculate_md5(self, file_path: str) -> str:
        """极速计算文件的 MD5 指纹"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # 每次读取 4096 字节，防止遇到几个 G 的文件撑爆内存
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    """提取年份和公司名"""

    def _extract_metadata(self, file_name: str) -> dict:
        year_match = re.search(r'(20\d{2})', file_name)
        company_match = re.search(r'^(.*?)(?:20\d{2})', file_name)

        # 获取原始匹配的字符串
        raw_company = company_match.group(1) if company_match else "未知"

        # 🌟 核心清洗逻辑：剔除字符串首尾的空格、全/半角冒号、破折号、下划线等标点符号
        # strip() 里面的字符就是我们要剔除的“黑名单”
        clean_company = raw_company.strip(" ：:_- \t")

        return {
            "year": year_match.group(1) if year_match else "未知",
            "company": clean_company,  # 存入清洗后的干净名称
            "source": file_name
        }

    def run_pipeline(self, pdf_path: str, original_filename: str = None, page_range: Optional[Tuple[int, int]] = None):
        """完整的端到端入库流程 (带哈希去重)"""
        try:
            path = Path(pdf_path)
            display_name = original_filename if original_filename else path.name

            # ==========================================
            # 🛡️ Step 0: 物理级指纹查重 (Hash Fingerprinting)
            # ==========================================
            logger.info(f"Step 0: 正在计算文件指纹并查重...")
            file_md5 = self._calculate_md5(pdf_path)

            db = SessionLocal()
            try:
                # 去数据库里查一查这个指纹有没有登记过
                existing_file = db.query(UploadedFile).filter(UploadedFile.file_hash == file_md5).first()
                if existing_file:
                    logger.warning(f"🚫 拦截重复文件！【{display_name}】(MD5: {file_md5}) 已于 {existing_file.upload_time} 入库。")
                    logger.warning("已自动跳过解析与向量化，防止数据库污染与 Token 浪费！")
                    return {"status": "skipped", "message": "文件已存在，无需重复入库"}
            finally:
                db.close() # 查完赶紧关门

            # ==========================================
            # A. 解析 PDF (Docling) - 🌟 内存保护：分块流式解析
            # ==========================================
            logger.info(f"Step 1: 启动内存安全模式 Parsing PDF {display_name}...")

            md_text = ""

            try:
                # 1. 先用极其轻量的 pypdf 偷看一下总页数
                reader = PdfReader(path)
                total_pages = len(reader.pages)
                logger.info(f"📄 检测到该文件共有 {total_pages} 页，准备切片解析...")

                # 2. 核心：每 30 页为一个批次，防止内存爆炸
                chunk_size = 30

                # 如果用户没有指定页码，我们就自己按批次循环
                if page_range is None:
                    for start_page in range(1, total_pages + 1, chunk_size):
                        end_page = min(start_page + chunk_size - 1, total_pages)
                        logger.info(f"⏳ 正在解析批次: 第 {start_page} ~ {end_page} 页...")

                        # 让 Docling 只处理这几十页
                        doc_result = self.converter.convert(path, page_range=(start_page, end_page))
                        # 把这部分转成 Markdown 拼接到总文本里
                        md_text += doc_result.document.export_to_markdown() + "\n\n"

                else:
                    # 如果用户本来就指定了小范围，就按用户的来
                    doc_result = self.converter.convert(path, page_range=page_range)
                    md_text = doc_result.document.export_to_markdown()

            except Exception as e:
                logger.error(f"❌ 分块解析失败，尝试退回全量解析: {e}")
                doc_result = self.converter.convert(path)
                md_text = doc_result.document.export_to_markdown()

            logger.info("✅ PDF 全部解析完毕，内存平稳释放！准备进行文本切分...")

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
            inserted_parent_ids = []  # 🌟 新增：准备一个小本本
            try:
                postgres_records = []
                for p_doc in safe_parent_docs:
                    record = ParentDocument(
                        id=p_doc.metadata["parent_id"],
                        content=p_doc.page_content,
                        meta_data=p_doc.metadata
                    )
                    postgres_records.append(record)
                    inserted_parent_ids.append(p_doc.metadata["parent_id"])  # 🌟 记下 ID

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
            # 🌟 新增：所有步骤都成功后，将文件指纹永久登记在案！
            db = SessionLocal()
            try:
                new_upload = UploadedFile(file_hash=file_md5, file_name=display_name)
                db.add(new_upload)
                db.commit()
                logger.info(f"✅ 文件指纹 {file_md5} 已登记，未来将自动拦截该文件的重复上传。")
            except Exception as e:
                db.rollback()
                logger.error(f"⚠️ 指纹登记失败: {str(e)}")
            finally:
                db.close()

            return {"status": "success", "message": "入库大闭环执行成功！"}


        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            # 🌟🌟🌟 新增：企业级分布式事务补偿机制 (Rollback Orphan Data) 🌟🌟🌟
            # 如果脚本崩溃了，并且刚才小本本上记了已经写入 Postgres 的 ID
            if 'inserted_parent_ids' in locals() and inserted_parent_ids:
                logger.warning("⚠️ 检测到后续流程(API/Milvus)崩溃，正在触发补偿事务...")
                logger.warning(f"🧹 正在从 PostgreSQL 擦除 {len(inserted_parent_ids)} 条孤儿父块数据，以保证双库一致性！")
                db_rollback = SessionLocal()
                try:
                    # 拿着小本本上的 ID，去数据库里把它们全删了！
                    db_rollback.query(ParentDocument).filter(
                        ParentDocument.id.in_(inserted_parent_ids)
                    ).delete(synchronize_session=False)
                    db_rollback.commit()
                    logger.info("✅ 补偿回滚成功！环境已恢复至入库前的纯净状态。")
                except Exception as rollback_err:
                    logger.error(f"❌ 灾难性故障：回滚 PostgreSQL 数据失败: {rollback_err}")
                finally:
                    db_rollback.close()
            raise e  # 最后还是要把原来的错误抛出来，让开发者知道为啥挂了



