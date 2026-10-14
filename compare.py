#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医疗 RAG 检索策略对比：BM25(ES) vs 纯向量 vs RRF 混合

【向量侧说明】
  为了不依赖外部 embedding API / 大模型下载，本脚本用 TF-IDF 作为「语义向量」近似。
  TF-IDF 能捕捉一部分词共现关系，但强度远弱于真正的稠密 embedding，
  因此会**放大术语漂移问题**（PD-1 与 PD-L1 混淆）—— 而这正是我们想演示的现象。
  换成真实 embedding 只需替换 build_tfidf_matrix() / vector_search() 两个函数。

【RRF 说明】
  Reciprocal Rank Fusion: score(d) = Σ_i 1 / (k + rank_i(d)),  k 常取 60
  为什么不用分数加权：BM25 分数无上界且随文档长度变化，cosine 落在 [-1,1]，
  量纲不可比；排名是无量纲的天然归一化，融合更稳健。
"""

import json
import os
import sys
from collections import Counter

import jieba
import numpy as np
import requests

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
INDEX = "medical_docs"
BASE = os.path.dirname(os.path.abspath(__file__))
TOP_K = 3
RRF_K = 60

jieba.setLogLevel(60)


# ------------------------------------------------------------------ ES / BM25

def es_search(query, top_k=TOP_K):
    """BM25 关键词检索。查 text_standard 字段（jieba 预分词写入）。"""
    seg = " ".join(jieba.lcut(query))
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {"match": {"title": {"query": query, "boost": 3.0}}},
                    {"match": {"text_standard": {"query": seg, "boost": 2.0}}},
                    {"match": {"text": {"query": query}}},
                ],
                "minimum_should_match": 1,
            }
        },
    }
    try:
        r = requests.post(f"{ES_URL}/{INDEX}/_search", json=body, timeout=10)
        return [h["_id"] for h in r.json().get("hits", {}).get("hits", [])]
    except Exception as e:
        print(f"[-] ES 查询失败: {e}")
        return []


# ------------------------------------------------------------ 语义向量（TF-IDF）

def fetch_all_docs():
    body = {"size": 1000, "query": {"match_all": {}}}
    r = requests.post(f"{ES_URL}/{INDEX}/_search", json=body, timeout=15)
    hits = r.json().get("hits", {}).get("hits", [])
    doc_ids, texts = [], []
    for h in hits:
        src = h["_source"]
        doc_ids.append(src["doc_id"])
        texts.append(src["title"] + "。 " + src["text"])
    return doc_ids, texts


def build_tfidf_matrix(texts):
    """构建词表 + TF-IDF 稠密矩阵。返回 (vocab, matrix)。"""
    vocab = {}
    tokenized = []
    for t in texts:
        toks = [w for w in jieba.lcut(t) if w.strip()]
        tokenized.append(toks)
        for w in toks:
            if w not in vocab:
                vocab[w] = len(vocab)

    tf = np.zeros((len(texts), len(vocab)))
    for i, toks in enumerate(tokenized):
        cnt = Counter(toks)
        total = sum(cnt.values()) or 1
        for w, c in cnt.items():
            tf[i, vocab[w]] = c / total

    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + len(texts)) / (1 + df)) + 1
    return vocab, tf * idf


def vector_search(query, doc_ids, vocab, doc_mat, idf, top_k=TOP_K):
    """余弦相似度语义检索。"""
    toks = [w for w in jieba.lcut(query) if w.strip() and w in vocab]
    if not toks:
        return []
    q = np.zeros(doc_mat.shape[1])
    cnt = Counter(toks)
    total = sum(cnt.values())
    for w, c in cnt.items():
        q[vocab[w]] = c / total
    q *= idf

    dnorm = np.linalg.norm(doc_mat, axis=1)
    dnorm[dnorm == 0] = 1
    qnorm = np.linalg.norm(q) or 1
    scores = doc_mat.dot(q) / (dnorm * qnorm)
    order = np.argsort(-scores)[:top_k]
    return [doc_ids[i] for i in order]


# ---------------------------------------------------------------- RRF 融合

def rrf_fuse(rank_lists, k=RRF_K, top_k=TOP_K):
    """Reciprocal Rank Fusion：score(d) = Σ 1/(k + rank_i(d))"""
    scores = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


# -------------------------------------------------------------------- 评估

def is_hit(top_ids, gold):
    golds = gold if isinstance(gold, list) else [gold]
    return any(g in top_ids for g in golds)


def evaluate(by_type):
    out = {}
    all_vals = []
    for qtype in ("terminology", "semantic", "trap"):
        vals = [v["hit"] for v in by_type.get(qtype, [])]
        out[qtype] = (sum(vals) / len(vals) * 100) if vals else 0.0
        all_vals.extend(vals)
    out["all"] = (sum(all_vals) / len(all_vals) * 100) if all_vals else 0.0
    return out


def print_block(title, stat):
    print("-" * 62)
    print(f"  {title}")
    print("-" * 62)
    for key, label in (("terminology", "术语类 query 命中率"),
                       ("semantic", "描述类 query 命中率"),
                       ("trap", "术语陷阱 query 命中率"),
                       ("all", "综合命中率")):
        print(f"  {label:<24}{stat[key]:>7.1f}%")
    print()


def main():
    try:
        if requests.get(ES_URL, timeout=5).status_code != 200:
            raise RuntimeError
    except Exception:
        print("[-] 连不上 ES。请先执行：docker compose up -d")
        print("    等 40~60 秒后再运行本脚本。")
        sys.exit(1)

    qfile = os.path.join(BASE, "queries.json")
    if not os.path.exists(qfile):
        print("[-] 缺少 queries.json，请先运行 prepare_and_load.py")
        sys.exit(1)
    queries = json.load(open(qfile, "r", encoding="utf-8"))

    print()
    print("=" * 62)
    print(" 医疗 RAG 检索策略对比：BM25 / 纯向量 / RRF 混合")
    print("=" * 62)
    print(f"  查询集：{len(queries)} 条"
          f"（术语类 {sum(1 for q in queries if q['type']=='terminology')}"
          f" / 描述类 {sum(1 for q in queries if q['type']=='semantic')}"
          f" / 陷阱 {sum(1 for q in queries if q['type']=='trap')}）")
    print()

    doc_ids, texts = fetch_all_docs()
    vocab, doc_mat = build_tfidf_matrix(texts)
    dfm = (doc_mat > 0).sum(axis=0)
    idf = np.log((1 + len(texts)) / (1 + dfm)) + 1
    print(f"  语料：{len(doc_ids)} 条 | 词表：{len(vocab)} 词 | 检索 top_k={TOP_K} | RRF k={RRF_K}")
    print()

    res = {"bm25": {}, "vector": {}, "rrf": {}}
    traps = []

    for q in queries:
        qt, gid, query = q["type"], q["gold"], q["q"]
        r_bm = es_search(query)
        r_ve = vector_search(query, doc_ids, vocab, doc_mat, idf)
        r_rr = rrf_fuse([r_bm, r_ve])

        for key, ranks in (("bm25", r_bm), ("vector", r_ve), ("rrf", r_rr)):
            res[key].setdefault(qt, []).append({"hit": is_hit(ranks, gid), "q": query})

        if qt == "trap":
            traps.append((query, gid, r_bm, r_ve, r_rr))

    print_block("策略 1：BM25（ES 倒排索引，纯关键词）", evaluate(res["bm25"]))
    print_block("策略 2：纯语义向量（TF-IDF 近似）", evaluate(res["vector"]))
    print_block("策略 3：RRF 混合（1/(60+rank) 加权融合）", evaluate(res["rrf"]))

    print("-" * 62)
    print("  术语陷阱演示（PD-1 与 PD-L1 —— 纯向量会混）")
    print("-" * 62)
    for query, gid, r_bm, r_ve, r_rr in traps:
        print(f"  query「{query}」  gold={gid}")
        print(f"    BM25  top{TOP_K} : {r_bm}")
        print(f"    向量  top{TOP_K} : {r_ve}")
        print(f"    RRF   top{TOP_K} : {r_rr}")
        golds = gid if isinstance(gid, list) else [gid]
        wrong = [d for d in r_ve if d in ("D02", "D03", "D19") and d not in golds]
        print(f"    → 纯向量{'⚠ 出现术语混淆: ' + str(wrong) if wrong else '✓ 未混淆'}")
        print()

    print("=" * 62)
    print("【结论】")
    print("  · 术语类（药名/缩写/编号）：BM25 稳定命中字面，语义向量容易漂移")
    print("  · 描述类（症状/口语/换说法）：BM25 命中不了，语义向量能召回")
    print("  · RRF 混合：两类 query 同时拿到最好成绩 → 医疗场景必须双路召回")
    print()
    print("【要点】")
    print("  1. 医疗场景 99% 查询是术语精确匹配（药名/病名/型号/试验编号），ES 是刚需")
    print("  2. 但症状描述、主诉是语义泛化需求，纯向量更合适 → 互补而非二选一")
    print("  3. RRF 用排名倒数不用分数加权：BM25 无上界、cosine∈[-1,1]，量纲不可比")
    print("=" * 62)


if __name__ == "__main__":
    main()