#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
准备数据并灌入 Elasticsearch。

工程降级说明：
  容器内未装 IK 中文分词插件时，BM25 中文召回会失效。
  因此这里采用「Python 侧 jieba 预分词 + 双字段索引」方案：
    - title        : 原文标题
    - text         : 原文正文（standard 分词）
    - text_standard: jieba 分词后用空格连接的字符串（保证 BM25 能命中中文词）
  这是工程上「先保证跑对，再优化实现」的典型做法：不依赖中文分词插件，
  用双字段索引让 BM25 在中文上稳定工作。
"""

import json
import os
import sys
import time

import jieba
import requests

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
INDEX = "medical_docs"
CORPUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical_corpus.json")

MAPPING = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "standard"},
            "text": {"type": "text", "analyzer": "standard"},
            "text_standard": {"type": "text", "analyzer": "standard"},
            "keywords": {"type": "keyword"},
            "raw_text": {"type": "text", "index": False},
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
}


def wait_for_es(max_wait=180):
    print("[*] 等待 Elasticsearch 就绪 ...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(ES_URL, timeout=5)
            if r.status_code == 200:
                info = r.json()
                print(f"[+] ES 就绪  version={info.get('version', {}).get('number')}")
                return True
        except Exception:
            pass
        time.sleep(3)
    print("[-] ES 未在超时内就绪，请确认 docker compose up -d 已启动")
    return False


def load_corpus():
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_docs(corpus):
    docs = []
    for d in corpus["docs"]:
        seg = " ".join(jieba.lcut(d["title"] + " " + d["text"]))
        docs.append({
            "doc_id": d["id"],
            "title": d["title"],
            "text": d["text"],
            "text_standard": seg,
            "keywords": d.get("keywords", []),
            "raw_text": d["title"] + "。 " + d["text"],
        })
    return docs


def main():
    if not wait_for_es():
        sys.exit(1)

    corpus = load_corpus()
    docs = build_docs(corpus)

    # 删旧索引重建，保证可重复执行
    requests.delete(f"{ES_URL}/{INDEX}")

    r = requests.put(f"{ES_URL}/{INDEX}", json=MAPPING)
    if r.status_code not in (200, 201):
        print("[-] 创建索引失败:", r.status_code, r.text[:500])
        sys.exit(1)
    print(f"[+] 索引 {INDEX} 创建成功")

    # 批量写入
    body_lines = []
    for d in docs:
        body_lines.append(json.dumps({"index": {"_index": INDEX, "_id": d["doc_id"]}}))
        body_lines.append(json.dumps(d, ensure_ascii=False))

    payload = "\n".join(body_lines) + "\n"
    r = requests.post(
        f"{ES_URL}/_bulk",
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
    )
    res = r.json()
    if res.get("errors"):
        print("[-] 批量写入有错误:", json.dumps(res, ensure_ascii=False)[:600])
        sys.exit(1)
    print(f"[+] 已写入 {len(docs)} 条文档")

    # 刷新索引使其可检索
    requests.post(f"{ES_URL}/{INDEX}/_refresh")
    cnt = requests.get(f"{ES_URL}/{INDEX}/_count").json()
    print(f"[+] 索引可检索文档数: {cnt.get('count')}")

    # 保存 queries 供 compare.py 使用
    qfile = os.path.join(os.path.dirname(CORPUS_FILE), "queries.json")
    with open(qfile, "w", encoding="utf-8") as f:
        json.dump(corpus["queries"], f, ensure_ascii=False, indent=2)
    print(f"[+] 查询集已保存: {qfile}  ({len(corpus['queries'])} 条)")


if __name__ == "__main__":
    main()