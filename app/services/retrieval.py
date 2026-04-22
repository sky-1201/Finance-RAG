import logging
import json
import dashscope
from typing import List, Optional
from langchain_milvus import Milvus
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from app.core.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    智能检索服务：实现了 标量过滤 -> 子块向量检索 -> 父块召回 -> 语义重排序 的完整 Pipeline。
    """

    def __init__(self):
        # 1. 配置全局 DashScope API Key
        dashscope.api_key = settings.DASHSCOPE_API_KEY

        # 2. 初始化 Embedding 模型
        self.embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )

        # 3. 连接到 Milvus
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=settings.COLLECTION_NAME,
            connection_args={"uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"}
        )

    def _retrieve_child_chunks(self, query: str, company: Optional[str], year: Optional[str], top_k: int = 10) -> List[
        Document]:
        """第一步：带 Metadata 过滤的子块查询"""
        # 强制只搜索子块
        expr_parts = ["doc_level == 'child'"]
        if company:
            expr_parts.append(f"company == '{company}'")
        if year:
            expr_parts.append(f"year == '{year}'")

        expr = " and ".join(expr_parts)
        logger.info(f"🔎 执行子块检索 | 表达式: {expr}")

        try:
            # 召回稍微多一点的数据 (top_k=10)，为后续的重排留出空间
            docs = self.vector_store.similarity_search(query=query, k=top_k, expr=expr)
            return docs
        except Exception as e:
            logger.error(f"❌ 子块检索失败: {str(e)}", exc_info=True)
            return []

    def _fetch_parent_chunks(self, child_docs: List[Document]) -> List[Document]:
        """第二步：根据子块的 parent_id 召回完整的父块"""
        if not child_docs:
            return []

        # 提取去重后的 parent_ids
        parent_ids = list(set([doc.metadata.get("parent_id") for doc in child_docs if doc.metadata.get("parent_id")]))

        if not parent_ids:
            return child_docs  # 如果没有父ID，降级返回子块

        # 构建查询表达式 (Milvus 的 IN 语法要求列表是 JSON 格式字符串或合法的 Python 列表表示)
        parent_ids_str = json.dumps(parent_ids)
        expr = f"doc_level == 'parent' and parent_id in {parent_ids_str}"

        logger.info(f"🔗 顺藤摸瓜：根据命中子块，提取了 {len(parent_ids)} 个父块 ID进行全文召回。")

        try:
            # 这里的 query 可以随便传，因为我们是靠 expr (ID匹配) 强制拉取数据的
            # k 设置为 parent_ids 的长度，确保全量召回
            parent_docs = self.vector_store.similarity_search(query="dummy", k=len(parent_ids), expr=expr)
            return parent_docs
        except Exception as e:
            logger.error(f"❌ 父块召回失败: {str(e)}", exc_info=True)
            return child_docs  # 失败时降级返回子块

    def _rerank_documents(self, query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
        """第三步：调用阿里 gte-rerank 模型进行重排序"""
        if not docs:
            return []

        logger.info(f"⚖️ 开始对 {len(docs)} 个完整父块进行 Rerank 重排序...")

        # 提取纯文本内容喂给重排模型
        doc_texts = [doc.page_content for doc in docs]

        try:
            # 调用 DashScope 的重排 API
            response = dashscope.TextReRank.call(
                model=dashscope.TextReRank.Models.gte_rerank,
                query=query,
                docs=doc_texts,
                top_n=top_n,
                return_documents=False
            )

            if response.status_code == 200:
                # 解析返回结果
                reranked_docs = []
                # response.output.results 是一个列表，包含排好序的 index 和 relevance_score
                for result in response.output.results:
                    original_index = result.index
                    score = result.relevance_score

                    # 取出原始 Document，并将重排分数写入 Metadata
                    doc = docs[original_index]
                    doc.metadata["rerank_score"] = score
                    reranked_docs.append(doc)

                logger.info(f"✅ Rerank 完成！提取 Top {top_n}。")
                return reranked_docs
            else:
                logger.error(f"❌ Rerank API 调用失败: 状态码 {response.status_code}, {response.message}")
                return docs[:top_n]  # 降级返回前 N 个

        except Exception as e:
            logger.error(f"❌ Rerank 过程发生异常: {str(e)}", exc_info=True)
            return docs[:top_n]

    def run_pipeline(self, query: str, company: Optional[str] = None, year: Optional[str] = None,
                     final_top_n: int = 3) -> List[Document]:
        """
        端到端检索接口：供最终 Agent / LLM 调用
        """
        try:
            # 1. 搜子块 (Top 10)
            child_docs = self._retrieve_child_chunks(query, company, year, top_k=10)

            # 2. 找父块
            parent_docs = self._fetch_parent_chunks(child_docs)

            # 3. 智能重排精选 (Top 3)
            final_docs = self._rerank_documents(query, parent_docs, top_n=final_top_n)

            return final_docs

        except Exception as e:
            logger.error(f"❌ 检索 Pipeline 崩溃: {str(e)}", exc_info=True)
            return []