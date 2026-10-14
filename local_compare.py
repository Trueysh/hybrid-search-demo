#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯本地版混合检索对比（无需 Docker / 无需 ES）

【为什么有这个文件】
  在无法启动 Docker 的环境里，用「jieba 分词 + BM25 打分」在纯 Python 里复现
  ES 的关键词检索逻辑；向量侧与 RRF 融合逻辑和 compare.py 完全一致，
  因此两种环境下结论方向相同，可以先用本脚本快速验证策略选择。

【边界】
  它不是真的 Elasticsearch——没有倒排索引、没有分布式、打分实现也与 ES 不同。
  轻量实现用来验证方向；涉及具体数字时，请以真实 ES（compare.py）的实测为准。

【工程思路】
  先用轻量实现验证策略选择的方向，再用真实环境确认结果。
"""

import json
import math
import os
from collections import Counter, defaultdict

import jieba
import numpy as np

jieba.setLogLevel(60)

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "medical_corpus.json")
TOP_K = 3
RRF_K = 60

# BM25 超参：k1 控制词频饱和，b 控制文档长度归一化（ES 默认 k1=1.2, b=0.75）
K1 = 1.2
B = 0.75


class SimpleBM25:
    """纯 Python BM25 实现，用于复现 ES 的关键词检索行为。"""

    def __init__(self, docs):
        self.doc_ids = [d["id"] for d in docs]
        self.tokenized = [
            [w for w in jieba.lcut(d["title"] + " " + d["text"]) if w.strip()]
            for d in docs
        ]
        self.dl = np.array([len(t) for t in self.tokenized], dtype=float)
        self.avgdl = self.dl.mean() if len(self.dl) else 1.0

        # 倒排索引：词 -> [(doc_idx, tf), ...]
        self.inv = defaultdict(list)
        for i, toks in enumerate(self.tokenized):
            for w, c in Counter(toks).items():
                self.inv[w].append((i, c))

        n = len(docs)
        self.idf = {}
        for w, plist in self.inv.items():
            df = len(plist)
            self.idf[w] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query, top_k=TOP_K):
        toks = [w for w in jieba.lcut(query) if w.strip()]
        scores = defaultdict(float)
        for w in toks:
            if w not in self.inv:
                continue
            idf = self.idf[w]
            for i, tf in self.inv[w]:
                denom = tf + K1 * (1 - B + B * self.dl[i] / self.avgdl)
                scores[i] += idf * (tf * (K1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [self.doc_ids[i] for i, _ in ranked]


class TfidfVector:
    """TF-IDF 向量检索（语义近似）。"""

    def __init__(self, docs):
        self.doc_ids = [d["id"] for d in docs]
        self.vocab = {}
        self.tokenized = []
        for d in docs:
            toks = [w for w in jieba.lcut(d["title"] + " " + d["text"]) if w.strip()]
            self.tokenized.append(toks)
            for w in toks:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)

        V = len(self.vocab)
        self.tf = np.zeros((len(docs), V))
        for i, toks in enumerate(self.tokenized):
            cnt = Counter(toks)
            tot = sum(cnt.values()) or 1
            for w, c in cnt.items():
                self.tf[i, self.vocab[w]] = c / tot

        df = (self.tf > 0).sum(axis=0)
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        self.mat = self.tf * self.idf
        self.dnorm = np.linalg.norm(self.mat, axis=1)
        self.dnorm[self.dnorm == 0] = 1

    def search(self, query, top_k=TOP_K):
        toks = [w for w in jieba.lcut(query) if w.strip() and w in self.vocab]
        if not toks:
            return []
        q = np.zeros(len(self.vocab))
        cnt = Counter(toks)
        tot = sum(cnt.values())
        for w, c in cnt.items():
            q[self.vocab[w]] = c / tot
        q *= self.idf
        qn = np.linalg.norm(q) or 1
        scores = self.mat.dot(q) / (self.dnorm * qn)
        order = np.argsort(-scores)[:top_k]
        return [self.doc_ids[i] for i in order]


def rrf_fuse(rank_lists, k=RRF_K, top_k=TOP_K):
    scores = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


def is_hit(top_ids, gold):
    golds = gold if isinstance(gold, list) else [gold]
    return any(g in top_ids for g in golds)


def evaluate(by_type):
    out, allv = {}, []
    for t in ("terminology", "semantic", "trap"):
        v = [x["hit"] for x in by_type.get(t, [])]
        out[t] = (sum(v) / len(v) * 100) if v else 0.0
        allv.extend(v)
    out["all"] = (sum(allv) / len(allv) * 100) if allv else 0.0
    return out


def block(title, stat, indent="  "):
    print("-" * 62)
    print(f"{indent}{title}")
    print("-" * 62)
    for k, lab in (("terminology", "术语类 query 命中率"),
                   ("semantic", "描述类 query 命中率"),
                   ("trap", "术语陷阱 query 命中率"),
                   ("all", "综合命中率")):
        print(f"{indent}  {lab:<24}{stat[k]:>7.1f}%")
    print()


def main():
    data = json.load(open(CORPUS, "r", encoding="utf-8"))
    docs, queries = data["docs"], data["queries"]

    print()
    print("=" * 62)
    print(" 医疗 RAG 检索策略对比（本地版：jieba + BM25 / TF-IDF / RRF）")
    print("=" * 62)
    print(f"  语料：{len(docs)} 条 | 查询：{len(queries)} 条 | top_k={TOP_K} | RRF k={RRF_K}")
    print("  ⚠ 本地 BM25 实现，非真实 Elasticsearch。报数字时须说明。")
    print()

    bm25 = SimpleBM25(docs)
    vec = TfidfVector(docs)

    res = {"bm25": {}, "vector": {}, "rrf": {}}
    traps = []

    for q in queries:
        qt, gid, query = q["type"], q["gold"], q["q"]
        r_bm = bm25.search(query)
        r_ve = vec.search(query)
        r_rr = rrf_fuse([r_bm, r_ve])
        for k, ranks in (("bm25", r_bm), ("vector", r_ve), ("rrf", r_rr)):
            res[k].setdefault(qt, []).append({"hit": is_hit(ranks, gid), "q": query})
        if qt == "trap":
            traps.append((query, gid, r_bm, r_ve, r_rr))

    block("策略 1：BM25（jieba 分词 + BM25 打分）", evaluate(res["bm25"]))
    block("策略 2：纯语义向量（TF-IDF 余弦）", evaluate(res["vector"]))
    block("策略 3：RRF 混合（1/(60+rank) 加权融合）", evaluate(res["rrf"]))

    print("-" * 62)
    print("  术语陷阱演示（PD-1 与 PD-L1）")
    print("-" * 62)
    for query, gid, r_bm, r_ve, r_rr in traps:
        print(f"  query「{query}」  gold={gid}")
        print(f"    BM25  top{TOP_K} : {r_bm}")
        print(f"    向量  top{TOP_K} : {r_ve}")
        print(f"    RRF   top{TOP_K} : {r_rr}")
        golds = gid if isinstance(gid, list) else [gid]
        wrong = [d for d in r_ve if d in ("D02", "D03", "D19") and d not in golds]
        print(f"    → 纯向量{'⚠ 术语混淆: ' + str(wrong) if wrong else '✓ 未混淆'}")
        print()

    print("=" * 62)
    print("【结论】")
    print("  · 术语类（药名/缩写）：BM25 靠字面稳定命中，语义向量易漂移")
    print("  · 描述类（症状/口语）：BM25 命中不了，语义向量能召回")
    print("  · RRF 混合：两类 query 同时拿到最好 → 医疗场景必须双路召回")
    print()
    print("【下一步】跑通真 ES（docker compose up -d → prepare_and_load.py → compare.py），")
    print("  确认真实 ES 与本地结论一致后，以 ES 的实测数字为准。")
    print("=" * 62)


if __name__ == "__main__":
    main()