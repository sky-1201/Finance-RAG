# 📈 Finance-RAG: 企业级金融投研 Agentic RAG 系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Milvus](https://img.shields.io/badge/Milvus-2.4-blueviolet)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Finance-RAG 是一个专为处理复杂金融财报、研报设计的企业级“检索增强生成（RAG）”与“多智能体（Agent）”结合的问答系统。针对大模型在垂直领域中常见的**“上下文碎片化”、“长文本召回率低”**以及**“财务数据计算幻觉”**三大痛点，提供了全链路的工程化解决方案。

---

## ✨ 核心亮点 (Core Highlights)

对于技术评审或面试官，本项目主要展示以下三大核心系统级设计：

### 1. 高可用检索引擎与计算下推 (High-Performance Retrieval & Computation Pushdown)
* **原生稀疏倒排索引 (防 OOM 机制)：** 为了应对企业级海量文本检索时的内存压力，系统采用了 Milvus 2.4 原生的 `SPARSE_INVERTED_INDEX`（稀疏倒排索引）来处理关键词特征。并在入库环节加入了低价值词过滤策略 (`drop_ratio_build`)，在保证检索准确率的同时，大幅降低了系统的内存占用与存储成本。
* **双路召回与计算下推：** 检索时，系统通过构建并发搜索请求 (`AnnSearchRequest`)，将 Dense (语义) 与 Sparse (关键词) 的双路检索，以及最终的 **RRF (倒数排序融合)** 计算，全部交由 Milvus 数据库的底层引擎并行执行。这种“计算下推”的设计避免了大量数据在数据库和应用层之间的来回传输，显著提升了系统的响应速度和并发吞吐量。

### 2. 存储解耦与高级架构 (Advanced RAG Architecture)
* **Parent-Child Chunking (父子块策略)：** 彻底抛弃传统单一向量切分方案。将具有完整语义的大段落（Parent）存入 PostgreSQL，将细粒度切分的子块（Child）向量化存入 Milvus。检索时命中子块，大模型阅读父块。**完美解决了 RAG 常见的语义截断和上下文丢失问题。**

### 3. 消除幻觉的 Agentic 工作流 (Agentic Workflow)
* **ReAct 范式智能体：** 摒弃死板的链式调用（Chain），基于 `LangGraph` 状态机重构问答链路。让大模型进行自主的“意图识别 -> 工具选择 -> 观察 -> 回答”。
* **代码沙盒执行 (Python REPL)：** 针对金融场景中“营收同比增长率”、“毛利率利润”等复杂数学计算，大模型极易产生严重幻觉。系统封装了 Python 代码执行沙盒工具，**大模型自主编写代码、自主查询上下文数据并执行计算**，确保了财务数据的 100% 绝对准确。

---

## 🛠 技术栈概览 (Tech Stack)

项目采用了现代化的前后端分离架构与云原生部署方案：

### 1. AI / 核心算法侧
* **编排框架:** `LangGraph`, `LangChain` (仅作底层工具类)
* **模型基座:** 兼容深求/智谱 API (文本生成), `BGE-m3` (稠密 Embedding), `BM25` (稀疏特征提取), `BGE-Reranker` (重排)
* **文档解析:** `Docling` / `PyMuPDF` (处理复杂图文混排版面)
* **底层数据计算:** `SciPy` (针对高维稀疏特征的 CSR 矩阵零拷贝提取)

### 2. 后端服务侧
* **Web 框架:** `FastAPI` (全异步非阻塞 IO, `BackgroundTasks` 异步防抖)
* **数据校验:** `Pydantic` V2
* **ORM 框架:** `SQLAlchemy`

### 3. 数据库 / 存储侧
* **向量数据库:** `Milvus 2.4` (HNSW 稠密索引 + SPARSE_INVERTED_INDEX 稀疏倒排索引 + 原生 RRF 引擎)
* **关系型数据库:** `PostgreSQL` (存储元数据、MD5 指纹防重、Parent Chunk 完整语义)

### 4. 前端与工程化
* **前端交互:** `Streamlit` (流式 SSE 输出响应)
* **部署运维:** `Docker`, `Docker Compose` 一键部署
* **系统评测:** 基于 `LLM-as-a-judge` 的自动化评测脚本

---

## 🏗 系统架构与核心逻辑 (Architecture)

### 1. 知识入库流 (Data Ingestion Pipeline)
为了保证企业级分布式双库数据强一致性，入库流程引入了**防重复校验**与**分布式事务补偿**：
1. **防重校验:** 上传 PDF 后，首先计算文件 MD5，在 PostgreSQL 中比对，避免同文件重复入库导致检索倾斜。
2. **多维特征提取:** 采用 Parent-Child 切分策略，同步提取稠密特征向量与稀疏特征矩阵（转化底层 CSR 格式）。
3. **事务补偿机制:** 将大文本落库 Postgres，将多路特征落库 Milvus。若 Milvus 写入失败，触发 Postgres 数据物理回滚 (Rollback Orphan Data)，杜绝脏数据。

### 2. 混合检索流 (Hybrid Retrieval Pipeline)
1. **Query 预处理:** 用户输入问题进行意图重写与关键词特征编码。
2. **底层计算下推与双路召回:** 构建并发搜索请求，利用 Milvus 原生引擎同时执行 Vector Search (语义相似度) 与 Sparse Search (关键词匹配)，并在数据库侧自动完成 RRF 倒数排序得分重塑。
3. **重排与溯源 (Reranking):** 将初步召回结果送入 Reranker 模型交叉打分，选出 Top-N 子块，并根据 `parent_id` 查出完整的父段落，拼接后送入大模型上下文。

### 3. Agent 调度流 (Agent Routing Pipeline)
用户提问 -> `LangGraph Router` 节点。
* 若判断为事实查询 -> 调用 `retriever_tool`。
* 若判断为财务计算 -> 提取检索数据 -> 调用 `finance_repl` 生成并执行 Python 代码。
* 汇总数据 -> 结合历史记忆 (Memory) -> 流式输出最终回答。

---

## 🚀 快速启动 (Quick Start)

本项目完全容器化，可通过 Docker Compose 一键拉起所有依赖环境。

### 1. 环境准备
确保本机已安装 `Docker` 与 `Docker Compose`。

### 2. 克隆与配置
```bash
git clone [https://github.com/your-username/Finance-RAG.git](https://github.com/your-username/Finance-RAG.git)
cd Finance-RAG
# 在根目录创建 .env 文件并填入你的 API Keys
cp .env.example .env
