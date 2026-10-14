# hybrid-search-demo

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.13-005571?style=flat&logo=elasticsearch&logoColor=white)](https://www.elastic.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

一个跑得起来的混合检索对比实验：在同一份医疗语料上，对比 **BM25（Elasticsearch）**、**纯向量（TF-IDF 余弦）**、**RRF 混合** 三种检索策略，并演示「术语陷阱」——为什么 `PD-1` 和 `PD-L1` 在语义向量里会被混为一谈。

## 背景

医疗 / 金融这类领域的查询分两类，任何单一策略都覆盖不了：

- **术语类**：药名、病名、器械型号、试验编号——99% 是精确匹配需求，字面不同就是不同东西
- **描述类**：症状、主诉、口语化描述——同一件事有几十说法，需要语义泛化

所以正确做法不是二选一，而是双路召回 + 融合。这个实验把这件事量化出来。

## 三种策略

| 策略 | 实现 | 适用 |
|---|---|---|
| BM25 | Elasticsearch 倒排索引（`match` + boost） | 术语类 query |
| 纯向量 | TF-IDF 稀疏向量 + 余弦相似度 | 描述类 query |
| RRF 混合 | Reciprocal Rank Fusion，`score = Σ 1/(60 + rank)` | 两类通吃 |

**为什么融合用排名倒数而不是分数加权**：BM25 分数无上界且随文档长度漂移，cosine 落在 [-1,1]，量纲不可比；排名是无量纲的天然归一化，融合更稳健。

## 快速开始

### 方式 A：真实 Elasticsearch（推荐）

```bash
# 1. 启动 ES 单节点（首次拉镜像后需等 40~60 秒）
docker compose up -d

# 2. 安装依赖
pip install -r requirements.txt

# 3. 建索引 + 灌数据（25 条语料，自动生成查询集）
python prepare_and_load.py

# 4. 跑三策略对比
python compare.py
```

### 方式 B：无 Docker 的轻量版

```bash
python local_compare.py
```

用纯 Python 复现 BM25 打分逻辑（jieba 分词 + 相同超参 k1=1.2 / b=0.75），向量侧与 RRF 逻辑和方式 A 完全一致。适合先验证策略方向；**具体数字请以真实 ES 的实测为准**。

### 可选：Kibana

```bash
docker compose --profile kb up -d   # http://localhost:5601
```

## 输出示例

```
============================================================
 医疗 RAG 检索策略对比：BM25 / 纯向量 / RRF 混合
============================================================
  查询集：14 条（术语类 / 描述类 / 术语陷阱）

------------------------------------------------------------
策略 1：BM25（ES 倒排索引，纯关键词）
  术语类 query 命中率：  XX.X%
  描述类 query 命中率：  XX.X%
  综合命中率：           XX.X%
------------------------------------------------------------
策略 2：纯语义向量（TF-IDF 近似）
  ...
------------------------------------------------------------
策略 3：RRF 混合（1/(60+rank) 加权融合）
  ...
------------------------------------------------------------
术语陷阱演示（PD-1 与 PD-L1 —— 纯向量会混）
  query「...」 gold=D02
    BM25  top3 : [...]
    向量  top3 : [...]   → ⚠ 出现术语混淆
    RRF   top3 : [...]   → ✓ 未混淆
============================================================
```

（`XX.X%` 为格式占位——在自己环境里跑出来的数字才是可信的。）

## 一个值得留意的工程细节

容器镜像里默认没有 IK 中文分词插件，BM25 对中文会失效。本项目的处理是：**Python 侧 jieba 预分词 + 双字段索引**（`text` 原文 + `text_standard` 预分词），保证 BM25 一定能命中中文词，不依赖 ES 插件。先保证跑对，再优化实现。

## 自定义语料

改 `medical_corpus.json` 即可：每条 doc 有 `id / title / text / keywords`，每条 query 标注 `gold`（期望命中的 doc id）与 `type`（`terminology` / `semantic` / `trap`）。可以换成任何垂直领域的语料，对比不同领域的术语密度差异。

## 目录结构

```
.
├── docker-compose.yml      # ES 8.13 单节点 + Kibana（profile: kb）
├── requirements.txt        # elasticsearch / jieba / numpy / requests
├── prepare_and_load.py     # 建索引 + 灌数据 + 生成查询集
├── compare.py              # 真实 ES 三策略对比
├── local_compare.py        # 无 Docker 的纯 Python 复现版
└── medical_corpus.json     # 25 条语料 + 14 条标注查询
```

## 向量侧的说明

向量侧用 TF-IDF 而非稠密 embedding，是为了零外部依赖（不调 embedding API、不下载模型）。TF-IDF 强度弱于真 embedding，会**放大**术语漂移现象——而这正是本实验想演示的效果。换成真实 embedding 只需替换 `compare.py` 里的 `build_tfidf_matrix()` 和 `vector_search()` 两个函数。

## License

MIT，见 [LICENSE](LICENSE)。
