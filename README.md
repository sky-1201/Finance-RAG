# 🚀 FinanceRAG: 企业级金融财报智能问答系统

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Docker](https://img.shields.io/badge/Docker-Compose-2496ED) ![Milvus](https://img.shields.io/badge/Milvus-2.4.0-blueviolet)

## 📖 项目简介
本项目是一个基于云原生架构（Docker 容器化）的企业级 RAG（检索增强生成）系统。专门针对百页以上的长文本金融财报（如《年度报告》）进行深度解析与精准问答。
系统实现了**计算层与存储层的彻底解耦**，并支持实时流式输出，具备高可用、易部署的生产级特性。

## ✨ 核心特性
- **双库架构，动静分离**：使用 PostgreSQL 存储十几万字的财报全量原文与防重日志，使用 Milvus 向量数据库存储高维特征索引，确保检索速度与数据一致性。
- **混合检索增强 (Hybrid Search)**：结合 BM25 稀疏检索（关键词精准匹配）与 Dense 稠密向量检索（语义理解），极大提升金融专业术语的召回准确率。
- **流式响应与长文本处理**：采用 Generator 机制切片处理超大 PDF 文件告别 OOM，问答环节支持类似 ChatGPT 的打字机流式输出体验。
- **开箱即用 (IaC)**：全量代码、环境依赖与网络拓扑均通过 `docker-compose` 编排，实现 `0` 配置秒级部署。

## 🛠️ 技术栈
- **前端交互**：Streamlit
- **后端引擎**：FastAPI, Uvicorn
- **核心算法**：LangChain, Rank-BM25, Jieba
- **数据存储**：Milvus (向量), PostgreSQL (关系型), MinIO (对象存储)
- **大模型接入**：阿里云通义千问 (DashScope API)

## 🚀 极速启动 (Quick Start)
确保本机已安装 Docker Desktop，在项目根目录执行：
```bash
# 1. 复制环境变量模板并填入大模型 API Key
cp .env.example .env

# 2. 一键拉起微服务全家桶
docker-compose up -d
