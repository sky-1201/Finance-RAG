import logging
import json
import dashscope
from typing import List, Optional
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from app.core.config import settings
from app.services.hybrid_search import HybridSearchEngine

# 🌟 关键：彻底抛弃 langchain_milvus，直接使用原生
from pymilvus import connections, Collection

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self):
        dashscope.api_key = settings.DASHSCOPE_API_KEY

        self.embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )

        # 🌟 彻底抛弃 LangChain Milvus，直接用纯原生连库！
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            logger.info("🔌 底层原生 PyMilvus 连接激活成功！")
        except Exception as e:
            logger.warning(f"⚠️ PyMilvus 连接复用提示: {e}")

        # 🌟 初始化原生 Collection 并加载到内存（Milvus 搜索前必须 load）
        self.collection = Collection(settings.COLLECTION_NAME)
        self.collection.load()
        logger.info("📦 Milvus 数据表已成功加载到内存，随时可以检索。")

    def _retrieve_child_chunks(self, query: str, company: Optional[str], year: Optional[str], top_k: int = 10) -> List[
        Document]:
        # 原生 JSON 字段过滤语法
        expr_parts = ['metadata["doc_level"] == "child"']
        if company:
            expr_parts.append(f'metadata["company"] == "{company}"')
        if year:
            expr_parts.append(f'metadata["year"] == "{year}"')

        expr = " and ".join(expr_parts)
        logger.info(f"🔎 执行原生子块检索 | 表达式: {expr}")

        try:
            query_vector = self.embeddings.embed_query(query)

            # 🌟 核心修复 1：每次检索前强制同步硬盘最新数据进内存！
            self.collection.load()

            # 🌟 原生向量检索
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "L2", "params": {}},
                limit=top_k,
                expr=expr,
                output_fields=["text", "metadata"]
            )

            docs = []
            # results[0] 对应的是我们传入的唯一一个查询向量的命中结果
            for hit in results[0]:
                text = hit.entity.get("text")
                meta = hit.entity.get("metadata", {})
                docs.append(Document(page_content=text, metadata=meta))

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
        expr = f"metadata['doc_level'] == 'parent' and metadata['parent_id'] in {parent_ids_str}"

        logger.info(f"🔗 顺藤摸瓜：提取了 {len(parent_ids)} 个完整父块。")

        try:
            # 🌟 原生标量查询（不需要向量计算，速度极快）
            results = self.collection.query(
                expr=expr,
                output_fields=["text", "metadata"],
                limit=len(parent_ids)
            )

            parent_docs = []
            for res in results:
                text = res.get("text")
                meta = res.get("metadata", {})
                parent_docs.append(Document(page_content=text, metadata=meta))

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
                documents=doc_texts,
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

    def _fetch_all_child_chunks_for_bm25(self, company: Optional[str], year: Optional[str]) -> List[dict]:
        """将特定公司/年份的所有子块拉入内存，构建全局 BM25 候选池"""
        expr_parts = ['metadata["doc_level"] == "child"']
        if company:
            expr_parts.append(f'metadata["company"] == "{company}"')
        if year:
            expr_parts.append(f'metadata["year"] == "{year}"')

        expr = " and ".join(expr_parts)

        try:
            # 使用标量 query 极速拉取所有符合条件的子块
            results = self.collection.query(
                expr=expr,
                output_fields=["text", "metadata"],
                limit=16384  # 设置一个足够大的值，确保拉出该财报所有子块
            )
            return results
        except Exception as e:
            logger.error(f"❌ 拉取全量子块构建 BM25 索引失败: {str(e)}")
            return []

    def run_pipeline(self, query: str, company: Optional[str] = None, year: Optional[str] = None,
                     final_top_n: int = 3) -> List[Document]:
        try:
            hybrid_engine = HybridSearchEngine()

            # =======================================================
            # 🟢 第一阶段：并行双路召回 (目标：子块)
            # =======================================================
            # 1. 向量路 (Dense)：捞取语义相关的 Top 60 子块
            logger.info("👉 启动路 1：向量检索...")
            dense_child_docs = self._retrieve_child_chunks(query, company, year, top_k=60)
            # 转换成 dict 格式以适配 RRF
            dense_results = [{"text": doc.page_content, "metadata": doc.metadata} for doc in dense_child_docs]

            # 2. 关键词路 (Sparse/BM25)：捞取字面匹配的 Top 30 子块
            logger.info("👉 启动路 2：全局内存 BM25 检索...")
            all_corpus_dicts = self._fetch_all_child_chunks_for_bm25(company, year)
            bm25_results = hybrid_engine.execute_bm25_search(
                query=query,
                document_pool=all_corpus_dicts,
                top_n=30
            )

            # =======================================================
            # 🟣 第二阶段：RRF 子块融合打分
            # =======================================================
            logger.info("👉 阶段 2：执行 RRF 双路子块融合...")
            fused_dicts = hybrid_engine.compute_rrf(dense_results, bm25_results)

            # 提取融合后得分最高的前 15 个“极品子块”
            top_fused_dicts = fused_dicts[:15]
            top_fused_docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in top_fused_dicts]

            # =======================================================
            # 🟠 第三阶段：顺藤摸瓜找父块 (防收敛黑洞)
            # =======================================================
            logger.info("👉 阶段 3：基于最优子块，提取完整父块...")
            # 因为我们提供的子块既有语义强的，又有关键词强的，映射出的父块质量极高
            parent_docs = self._fetch_parent_chunks(top_fused_docs)

            # =======================================================
            # 🔴 第四阶段：大模型终极重排
            # =======================================================
            final_docs = self._rerank_documents(query, parent_docs, top_n=final_top_n)

            return final_docs

        except Exception as e:
            logger.error(f"❌ 检索 Pipeline 崩溃: {str(e)}", exc_info=True)
            return []