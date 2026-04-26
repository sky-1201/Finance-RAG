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
    def __init__(self):
        dashscope.api_key = settings.DASHSCOPE_API_KEY

        self.embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )

        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=settings.COLLECTION_NAME,
            connection_args={"uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"}
        )

    def _retrieve_child_chunks(self, query: str, company: Optional[str], year: Optional[str], top_k: int = 10) -> List[Document]:
        expr_parts = ["doc_level == 'child'"]
        if company:
            expr_parts.append(f"company == '{company}'")
        if year:
            expr_parts.append(f"year == '{year}'")

        expr = " and ".join(expr_parts)
        logger.info(f"🔎 执行子块检索 | 表达式: {expr}")

        try:
            docs = self.vector_store.similarity_search(query=query, k=top_k, expr=expr)
            return docs
        except Exception as e:
            logger.error(f"❌ 子块检索失败: {str(e)}", exc_info=True)
            return []

    def _fetch_parent_chunks(self, child_docs: List[Document]) -> List[Document]:
        if not child_docs:
            return []

        parent_ids = list(set([doc.metadata.get("parent_id") for doc in child_docs if doc.metadata.get("parent_id")]))

        if not parent_ids:
            return child_docs

        parent_ids_str = json.dumps(parent_ids)
        expr = f"doc_level == 'parent' and parent_id in {parent_ids_str}"

        logger.info(f"🔗 顺藤摸瓜：根据命中子块，提取了 {len(parent_ids)} 个父块 ID进行全文召回。")

        try:
            parent_docs = self.vector_store.similarity_search(query="dummy", k=len(parent_ids), expr=expr)
            return parent_docs
        except Exception as e:
            logger.error(f"❌ 父块召回失败: {str(e)}", exc_info=True)
            return child_docs

    def _rerank_documents(self, query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
        if not docs:
            return []

        logger.info(f"⚖️ 开始对 {len(docs)} 个完整父块进行 Rerank 重排序...")
        doc_texts = [doc.page_content for doc in docs]

        try:
            response = dashscope.TextReRank.call(
                model=dashscope.TextReRank.Models.gte_rerank,
                query=query,
                docs=doc_texts,
                top_n=top_n,
                return_documents=False
            )

            if response.status_code == 200:
                reranked_docs = []
                for result in response.output.results:
                    original_index = result.index
                    score = result.relevance_score
                    doc = docs[original_index]
                    doc.metadata["rerank_score"] = score
                    reranked_docs.append(doc)

                logger.info(f"✅ Rerank 完成！提取 Top {top_n}。")
                return reranked_docs
            else:
                logger.error(f"❌ Rerank API 调用失败: 状态码 {response.status_code}, {response.message}")
                return docs[:top_n]

        except Exception as e:
            logger.error(f"❌ Rerank 过程发生异常: {str(e)}", exc_info=True)
            return docs[:top_n]

    def run_pipeline(self, query: str, company: Optional[str] = None, year: Optional[str] = None, final_top_n: int = 3) -> List[Document]:
        try:
            child_docs = self._retrieve_child_chunks(query, company, year, top_k=10)
            parent_docs = self._fetch_parent_chunks(child_docs)
            final_docs = self._rerank_documents(query, parent_docs, top_n=final_top_n)
            return final_docs
        except Exception as e:
            logger.error(f"❌ 检索 Pipeline 崩溃: {str(e)}", exc_info=True)
            return []