import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict


class HybridSearchEngine:
    def __init__(self, k_dense: int = 60, k_bm25: int = 40):
        # 初始化双路各自召回的数量 (可以多捞一点，反正后面会做 RRF 融合)
        self.k_dense = k_dense
        self.k_bm25 = k_bm25

    def _tokenize(self, text: str) -> List[str]:
        """使用结巴分词对中文进行精准切词"""
        return list(jieba.cut_for_search(text))

    def compute_rrf(self, dense_results: List[Dict], bm25_results: List[Dict], rrf_k: int = 60) -> List[Dict]:
        """
        核心算法：倒数排序融合 (Reciprocal Rank Fusion)
        参数:
            dense_results: 向量检索回来的带 text 的字典列表
            bm25_results: BM25 检索回来的带 text 的字典列表
            rrf_k: 缓解常数，业界通常设置为 60
        """
        # 用一个字典来记录每个文本块的 RRF 得分
        # 我们用 chunk 的文本本身 (或 ID) 作为主键去重
        rrf_scores = {}
        combined_results_map = {}

        # 1. 计算 Dense 向量的 RRF 得分
        for rank, doc in enumerate(dense_results):
            text = doc.get("text", "")
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
                combined_results_map[text] = doc
            # RRF 公式核心：1 / (rank + 1 + rrf_k)
            rrf_scores[text] += 1.0 / (rank + 1 + rrf_k)

        # 2. 计算 BM25 的 RRF 得分
        for rank, doc in enumerate(bm25_results):
            text = doc.get("text", "")
            if text not in rrf_scores:
                rrf_scores[text] = 0.0
                combined_results_map[text] = doc
            rrf_scores[text] += 1.0 / (rank + 1 + rrf_k)

        # 3. 按最终的 RRF 得分降序排列
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. 组装返回结果
        final_list = []
        for text, score in sorted_results:
            doc_data = combined_results_map[text]
            doc_data["rrf_score"] = score  # 记录一下分数方便我们调试
            final_list.append(doc_data)

        return final_list

    def execute_bm25_search(self, query: str, document_pool: List[Dict], top_n: int = 15) -> List[Dict]:
        """
        在内存中对全量或局部文档池执行极速 BM25 检索
        """
        if not document_pool:
            return []

        # 把文档池中的文本抽出来进行分词
        tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in document_pool]

        # 构建 BM25 倒排索引
        bm25 = BM25Okapi(tokenized_corpus)

        # 对查询词进行相同的分词
        tokenized_query = self._tokenize(query)

        # 批量打分
        doc_scores = bm25.get_scores(tokenized_query)

        # 获取 Top-N 索引
        top_indices = np.argsort(doc_scores)[::-1][:top_n]

        # 返回对应的原始文档 (过滤掉得分为0的垃圾匹配)
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                results.append(document_pool[idx])

        return results