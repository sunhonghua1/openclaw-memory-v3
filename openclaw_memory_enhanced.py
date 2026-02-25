#!/usr/bin/env python3
"""
OpenClaw Enhanced Memory System V3.0
真向量语义搜索 + BM25 混合检索 + 多供应商 Embedding

升级要点（V2 → V3）：
1. Jaccard 词袋匹配 → 余弦向量语义搜索（真 Embedding）
2. 集成 DashScope/Google/Jina 三供应商自动 fallback
3. 向量缓存（避免重复调用 API）
4. 保留 BM25 做混合检索
"""

import json
import re
import time
import math
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field

# 导入多供应商 Embedding
from embedding_provider import (
    MultiProviderEmbedding, cosine_similarity, EmbeddingResult
)


# ========== 向量缓存 ==========

class VectorCache:
    """
    向量缓存管理器
    将已计算的向量存入 JSON 文件，避免重复调用 Embedding API

    缓存策略：基于文本内容的 hash 做键
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        self._load()

    def _load(self):
        if Path(self.cache_path).exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save(self):
        try:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 向量缓存保存失败: {e}")

    @staticmethod
    def _text_key(text: str) -> str:
        """生成文本的缓存键（简单 hash）"""
        # 取前 200 字符 + 长度作为键，避免超长键
        return f"{hash(text[:200])}_{len(text)}"

    def get(self, text: str) -> Optional[List[float]]:
        key = self._text_key(text)
        return self.cache.get(key)

    def put(self, text: str, vector: List[float]):
        key = self._text_key(text)
        self.cache[key] = vector
        # 缓存超过 5000 条时清理最早的（简单 FIFO）
        if len(self.cache) > 5000:
            keys = list(self.cache.keys())
            for old_key in keys[:1000]:
                del self.cache[old_key]
        self._save()

    def batch_get(self, texts: List[str]) -> Tuple[List[str], List[int], List[List[float]]]:
        """
        批量获取：返回未缓存的文本列表、其索引、以及已缓存的向量

        Returns:
            (uncached_texts, uncached_indices, cached_vectors_at_positions)
        """
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        return uncached_texts, uncached_indices, results


# ========== 搜索结果 ==========

@dataclass
class SearchResult:
    """搜索结果"""
    content: str
    score: float
    source: str
    timestamp: str
    metadata: Dict = field(default_factory=dict)


# ========== 混合搜索引擎 V3 ==========

class HybridSearchEngine:
    """
    混合搜索引擎 V3

    核心升级：
    - 向量语义搜索：使用真实 Embedding（余弦相似度）
    - BM25 关键词搜索：保留用于精确匹配
    - 混合融合：70% 向量 + 30% BM25
    """

    def __init__(self, embedder: MultiProviderEmbedding,
                 vector_cache: VectorCache):
        self.embedder = embedder
        self.vector_cache = vector_cache
        self.documents = []
        self.doc_vectors = []  # 每个文档对应的向量
        self.index = {}        # BM25 倒排索引
        self.idf_scores = {}

    def add_document(self, doc_id: str, content: str,
                     metadata: Dict = None, vector: List[float] = None):
        """添加文档（如果没有向量则自动生成）"""
        doc = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "tokens": self._tokenize(content),
            "timestamp": datetime.now().isoformat()
        }
        self.documents.append(doc)

        # 获取或生成向量
        if vector is not None:
            self.doc_vectors.append(vector)
        else:
            cached = self.vector_cache.get(content)
            if cached is not None:
                self.doc_vectors.append(cached)
            else:
                try:
                    result = self.embedder.embed([content])
                    vec = result.vectors[0]
                    self.doc_vectors.append(vec)
                    self.vector_cache.put(content, vec)
                except Exception as e:
                    # Embedding 失败时用零向量占位，不影响 BM25
                    print(f"⚠️ Embedding 失败 (doc={doc_id}): {e}")
                    self.doc_vectors.append([])

        # 更新 BM25 索引
        self._update_index(doc)
        self.idf_scores = {}  # 重置 IDF 缓存

    def _tokenize(self, text: str) -> List[str]:
        """分词（支持中英文）"""
        text_lower = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text_lower.split()
        # 中文按字符拆分（可升级为 jieba）
        chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)
        for chars in chinese_chars:
            # 按 2-gram 拆分以提升中文匹配精度
            for i in range(len(chars)):
                tokens.append(chars[i])
                if i < len(chars) - 1:
                    tokens.append(chars[i:i+2])
        return tokens

    def _update_index(self, doc: Dict):
        for token in set(doc["tokens"]):
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(doc["id"])

    def _calculate_idf(self):
        total_docs = max(len(self.documents), 1)
        for token, doc_ids in self.index.items():
            self.idf_scores[token] = math.log(total_docs / len(doc_ids))

    def _bm25_score(self, query_tokens: List[str], doc: Dict) -> float:
        """BM25 评分"""
        score = 0.0
        doc_tokens = doc["tokens"]
        doc_len = len(doc_tokens)
        avg_len = sum(len(d["tokens"]) for d in self.documents) / max(len(self.documents), 1)
        k1, b = 1.5, 0.75

        for token in query_tokens:
            if token in doc_tokens:
                tf = doc_tokens.count(token)
                idf = self.idf_scores.get(token, 0)
                norm = 1 - b + b * (doc_len / max(avg_len, 1))
                score += idf * (tf * (k1 + 1)) / (tf + k1 * norm)

        return score

    def hybrid_search(self, query: str, top_k: int = 5,
                      vector_weight: float = 0.7) -> List[SearchResult]:
        """
        混合搜索

        Args:
            query: 查询文本
            top_k: 返回 top-k 结果
            vector_weight: 向量搜索权重（0-1），剩余为 BM25 权重
        """
        if not self.documents:
            return []

        # 计算 IDF
        if not self.idf_scores:
            self._calculate_idf()

        # 获取查询向量
        query_vector = None
        try:
            cached = self.vector_cache.get(query)
            if cached is not None:
                query_vector = cached
            else:
                result = self.embedder.embed([query])
                query_vector = result.vectors[0]
                self.vector_cache.put(query, query_vector)
        except Exception as e:
            print(f"⚠️ 查询 Embedding 失败: {e}，将仅使用 BM25")
            vector_weight = 0.0  # 降级为纯 BM25

        query_tokens = self._tokenize(query)
        results = []

        for i, doc in enumerate(self.documents):
            # BM25 分数
            bm25 = self._bm25_score(query_tokens, doc)

            # 向量分数
            vec_score = 0.0
            if query_vector and i < len(self.doc_vectors) and self.doc_vectors[i]:
                vec_score = cosine_similarity(query_vector, self.doc_vectors[i])
                # 余弦相似度范围 [-1, 1]，归一化到 [0, 1]
                vec_score = (vec_score + 1) / 2

            # 混合分数
            final_score = vector_weight * vec_score + (1 - vector_weight) * bm25

            results.append(SearchResult(
                content=doc["content"],
                score=final_score,
                source=doc["id"],
                timestamp=doc["timestamp"],
                metadata=doc.get("metadata", {})
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "unique_tokens": len(self.index),
            "cached_vectors": len(self.vector_cache.cache),
            "avg_doc_length": (
                sum(len(d["tokens"]) for d in self.documents) /
                max(len(self.documents), 1)
            ),
            "embedding_provider": self.embedder.get_stats()
        }


# ========== 增强版记忆系统 V3 ==========

class EnhancedMemoryCore:
    """
    增强版记忆核心 V3

    V2 → V3 升级：
    1. Jaccard 词袋 → 真向量语义搜索
    2. 多供应商 Embedding（DashScope/Google/Jina）
    3. 向量缓存（避免重复 API 调用）
    """

    def __init__(self, storage_path: str = "/root/.openclaw/memory/openclaw_memory_v3.json",
                 config_dir: str = None):
        self.storage_path = storage_path

        # 确定配置目录
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))

        config_path = os.path.join(config_dir, "embedding_config.json")
        cache_path = os.path.join(
            os.path.dirname(storage_path), "vector_cache.json"
        )

        # 初始化 Embedding 管理器
        self.embedder = MultiProviderEmbedding(config_path=config_path)

        # 初始化向量缓存
        self.vector_cache = VectorCache(cache_path)

        # 分类字典（保留 V2 的结构）
        self.context = {
            "session": {
                "current_id": None,
                "start_time": None,
                "message_count": 0
            },
            "user_profile": {
                "preferences": {},
                "expertise": {},
                "history_summary": deque(maxlen=50)
            },
            "knowledge_base": {
                "code_snippets": {},
                "documents": {},
                "concepts": {}
            },
            "tasks": {
                "active": deque(maxlen=10),
                "completed": deque(maxlen=20)
            },
            "conversation_log": deque(maxlen=100)
        }

        # 混合搜索引擎（V3 核心！）
        self.search_engine = HybridSearchEngine(
            embedder=self.embedder,
            vector_cache=self.vector_cache
        )

        # 统计
        self.stats = {
            "searches": 0,
            "hits": 0,
            "token_saved": 0
        }

        self.load()
        self._rebuild_search_index()

    def load(self):
        """加载记忆"""
        if Path(self.storage_path).exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for key in ["user_profile", "tasks", "conversation_log"]:
                    if key in data:
                        if key == "conversation_log":
                            self.context[key] = deque(data[key], maxlen=100)
                        else:
                            for subkey, value in data[key].items():
                                if isinstance(value, list):
                                    max_len = 50 if "history" in subkey else 20
                                    self.context[key][subkey] = deque(
                                        value, maxlen=max_len
                                    )

                print("✅ 记忆加载成功")
            except Exception as e:
                print(f"⚠️ 加载失败: {e}")

    def save(self):
        """保存记忆"""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            serializable = self._to_serializable(self.context)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False

    def _to_serializable(self, obj):
        if isinstance(obj, deque):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(item) for item in obj]
        return obj

    def _rebuild_search_index(self):
        """重建搜索索引（启动时执行，使用缓存向量避免重复 API 调用）"""
        print("🔄 重建搜索索引...")
        indexed = 0

        for i, msg in enumerate(self.context["conversation_log"]):
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if content:
                    self.search_engine.add_document(
                        doc_id=f"conversation_{i}",
                        content=content,
                        metadata={
                            "role": msg.get("role"),
                            "timestamp": msg.get("timestamp")
                        }
                    )
                    indexed += 1

        for key, value in self.context["knowledge_base"].items():
            if isinstance(value, dict):
                for item_id, item_content in value.items():
                    content = str(item_content)
                    if content:
                        self.search_engine.add_document(
                            doc_id=f"knowledge_{key}_{item_id}",
                            content=content,
                            metadata={"category": key}
                        )
                        indexed += 1

        print(f"✅ 索引重建完成: {indexed} 条文档")

    # ========== 核心功能 ==========

    def smart_recall(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        智能回忆（V3 核心）

        使用真向量语义搜索 + BM25 混合检索
        能理解同义词和语义相近的表述
        """
        self.stats["searches"] += 1
        results = self.search_engine.hybrid_search(query, top_k=max_results)

        if results:
            self.stats["hits"] += 1
            total_size = sum(
                len(str(msg)) for msg in self.context["conversation_log"]
            )
            retrieved_size = sum(len(r.content) for r in results)
            self.stats["token_saved"] += (total_size - retrieved_size) // 4

        return [
            {
                "content": r.content,
                "score": r.score,
                "source": r.source,
                "timestamp": r.timestamp
            }
            for r in results
        ]

    def add_memory(self, content: str, category: str = "general",
                   metadata: Dict = None):
        """添加新记忆（自动嵌入 + 索引）"""
        timestamp = datetime.now().isoformat()

        self.context["conversation_log"].append({
            "content": content,
            "category": category,
            "timestamp": timestamp,
            "metadata": metadata or {}
        })

        doc_id = f"{category}_{len(self.context['conversation_log'])}"
        self.search_engine.add_document(doc_id, content, metadata)

        self.save()

    def get_relevant_context(self, current_query: str,
                             max_tokens: int = 500) -> str:
        """获取相关上下文（替代加载全部历史）"""
        relevant = self.smart_recall(current_query, max_results=3)

        if not relevant:
            return "（无相关历史记录）"

        parts = ["=== 相关记忆 ==="]
        current_tokens = 0

        for mem in relevant:
            text = f"[{mem['source']}] {mem['content'][:200]}"
            tokens = len(text) // 4
            if current_tokens + tokens > max_tokens:
                break
            parts.append(text)
            current_tokens += tokens

        return "\n".join(parts)

    def get_memory_stats(self) -> Dict:
        search_stats = self.search_engine.get_stats()
        return {
            **search_stats,
            "total_conversations": len(self.context["conversation_log"]),
            "active_tasks": len(self.context["tasks"]["active"]),
            "search_efficiency": {
                "total_searches": self.stats["searches"],
                "successful_hits": self.stats["hits"],
                "hit_rate": (
                    f"{self.stats['hits'] / self.stats['searches'] * 100:.1f}%"
                    if self.stats["searches"] > 0 else "0%"
                ),
                "estimated_tokens_saved": self.stats["token_saved"]
            }
        }

    def print_stats(self):
        stats = self.get_memory_stats()
        print("\n" + "=" * 60)
        print("📊 增强版记忆系统 V3 统计")
        print("=" * 60)
        print(f"💾 存储统计:")
        print(f"  - 文档总数: {stats['total_documents']}")
        print(f"  - 缓存向量: {stats['cached_vectors']}")
        print(f"  - 对话条数: {stats['total_conversations']}")
        print(f"\n🔍 搜索效率:")
        print(f"  - 搜索次数: {stats['search_efficiency']['total_searches']}")
        print(f"  - 命中率: {stats['search_efficiency']['hit_rate']}")
        print(f"  - 估算节省 Token: {stats['search_efficiency']['estimated_tokens_saved']:,}")
        print(f"\n🌐 Embedding 供应商:")
        for p in stats["embedding_provider"]["providers"]:
            status = "✅" if p["available"] else "❌"
            print(f"  {status} {p['name']} ({p['model']})")
        print("=" * 60 + "\n")


# ========== 演示 ==========

if __name__ == "__main__":
    print("🚀 OpenClaw 增强记忆系统 V3 演示\n")

    memory = EnhancedMemoryCore(
        storage_path="/tmp/test_memory_v3.json",
        config_dir=os.path.dirname(os.path.abspath(__file__))
    )

    # 添加记忆
    print("📝 添加历史记忆...")
    memory.add_memory("用户喜欢简洁的代码风格，不喜欢过多注释", category="preference")
    memory.add_memory("正在开发一个Python量化交易机器人", category="project")
    memory.add_memory("讨论了如何优化OpenClaw的Token消耗", category="conversation")
    memory.add_memory("ICLR论文截止日期是2026年3月", category="task")
    print("✅ 记忆已添加\n")

    # 关键测试：语义搜索 vs 词袋匹配
    print("=" * 60)
    print("🧪 关键测试：用不同的词搜索相同含义")
    print("=" * 60)

    test_queries = [
        ("编程规范", "应匹配'代码风格'（语义相近但词不同）"),
        ("量化策略", "应匹配'量化交易机器人'"),
        ("论文进度", "应匹配'ICLR论文截止日期'"),
    ]

    for query, expected in test_queries:
        print(f"\n🔍 查询: '{query}' — {expected}")
        results = memory.smart_recall(query, max_results=2)
        for r in results:
            print(f"  [{r['score']:.4f}] {r['content'][:60]}...")

    # 统计
    print()
    memory.print_stats()
