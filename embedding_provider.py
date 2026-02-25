#!/usr/bin/env python3
"""
多供应商 Embedding 模块
支持 DashScope (text-embedding-v4) / Google / Jina AI
自动 fallback 策略：DashScope → Google → Jina

零重型依赖：仅使用 Python 标准库的 urllib
"""

import json
import urllib.request
import urllib.error
import time
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Embedding 结果"""
    vectors: List[List[float]]
    provider: str
    model: str
    dimensions: int
    token_usage: int


class EmbeddingProvider:
    """Embedding 供应商基类"""

    def __init__(self, api_key: str, model: str, dimensions: int):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.name = "base"
        self._healthy = True
        self._last_error_time = 0
        # 错误后 60 秒内不再重试同一个 provider
        self._cooldown_seconds = 60

    @property
    def is_available(self) -> bool:
        """检查此 provider 是否可用（冷却期外）"""
        if not self.api_key:
            return False
        if not self._healthy:
            if time.time() - self._last_error_time > self._cooldown_seconds:
                self._healthy = True  # 冷却期过后重新尝试
            else:
                return False
        return True

    def _mark_failed(self):
        self._healthy = False
        self._last_error_time = time.time()

    def _mark_success(self):
        self._healthy = True

    def embed(self, texts: List[str]) -> EmbeddingResult:
        raise NotImplementedError


class DashScopeEmbedding(EmbeddingProvider):
    """
    阿里云 DashScope Embedding
    模型：text-embedding-v4
    免费额度：100 万 tokens（至 2026/05/23）
    """

    def __init__(self, api_key: str, model: str = "text-embedding-v4",
                 dimensions: int = 1024):
        super().__init__(api_key, model, dimensions)
        self.name = "dashscope"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def embed(self, texts: List[str]) -> EmbeddingResult:
        """调用 DashScope Embedding API（OpenAI 兼容格式）"""
        payload = json.dumps({
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self.dimensions
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            vectors = [item["embedding"] for item in data["data"]]
            usage = data.get("usage", {}).get("total_tokens", 0)

            self._mark_success()
            return EmbeddingResult(
                vectors=vectors,
                provider=self.name,
                model=self.model,
                dimensions=len(vectors[0]) if vectors else self.dimensions,
                token_usage=usage
            )
        except Exception as e:
            self._mark_failed()
            raise RuntimeError(f"[DashScope] Embedding 失败: {e}")


class GoogleEmbedding(EmbeddingProvider):
    """
    Google Gemini Embedding
    模型：gemini-embedding-001（text-embedding-004 已于 2026/01 弃用）
    免费额度：充足（Gemini API 内含）
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-001",
                 dimensions: int = 768):
        super().__init__(api_key, model, dimensions)
        self.name = "google"

    def embed(self, texts: List[str]) -> EmbeddingResult:
        """调用 Google Embedding API"""
        vectors = []
        total_tokens = 0

        for text in texts:
            payload = json.dumps({
                "model": f"models/{self.model}",
                "content": {
                    "parts": [{"text": text}]
                }
            }).encode("utf-8")

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"models/{self.model}:embedContent"
                f"?key={self.api_key}"
            )

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                vector = data["embedding"]["values"]
                vectors.append(vector)
                total_tokens += len(text) // 4  # 粗略估算
            except Exception as e:
                self._mark_failed()
                raise RuntimeError(f"[Google] Embedding 失败: {e}")

        self._mark_success()
        return EmbeddingResult(
            vectors=vectors,
            provider=self.name,
            model=self.model,
            dimensions=len(vectors[0]) if vectors else self.dimensions,
            token_usage=total_tokens
        )


class JinaEmbedding(EmbeddingProvider):
    """
    Jina AI Embedding
    模型：jina-embeddings-v3
    免费额度：1000 万 tokens/月
    """

    def __init__(self, api_key: str, model: str = "jina-embeddings-v3",
                 dimensions: int = 1024):
        super().__init__(api_key, model, dimensions)
        self.name = "jina"
        self.base_url = "https://api.jina.ai/v1"

    def embed(self, texts: List[str]) -> EmbeddingResult:
        """调用 Jina Embedding API（OpenAI 兼容格式）"""
        payload = json.dumps({
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            vectors = [item["embedding"] for item in data["data"]]
            usage = data.get("usage", {}).get("total_tokens", 0)

            self._mark_success()
            return EmbeddingResult(
                vectors=vectors,
                provider=self.name,
                model=self.model,
                dimensions=len(vectors[0]) if vectors else self.dimensions,
                token_usage=usage
            )
        except Exception as e:
            self._mark_failed()
            raise RuntimeError(f"[Jina] Embedding 失败: {e}")


class MultiProviderEmbedding:
    """
    多供应商 Embedding 管理器
    自动 fallback：DashScope → Google → Jina
    """

    def __init__(self, config_path: str = None, config: Dict = None):
        """
        Args:
            config_path: embedding_config.json 的路径
            config: 直接传入配置字典（优先于 config_path）
        """
        if config:
            self.config = config
        elif config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            raise ValueError("必须提供 config_path 或 config")

        self.providers: List[EmbeddingProvider] = []
        self.current_provider = None
        self._total_tokens_used = 0
        self._init_providers()

    def _init_providers(self):
        """按优先级初始化供应商"""
        provider_map = {
            "dashscope": DashScopeEmbedding,
            "google": GoogleEmbedding,
            "jina": JinaEmbedding
        }

        # 按配置的优先级排序
        primary = self.config.get("primary", "dashscope")
        providers_config = self.config.get("providers", {})

        # 主 provider 排在最前
        ordered_names = [primary]
        for name in providers_config:
            if name != primary:
                ordered_names.append(name)

        for name in ordered_names:
            if name in providers_config and name in provider_map:
                cfg = providers_config[name]
                api_key = cfg.get("api_key", "")
                if not api_key:
                    continue
                provider = provider_map[name](
                    api_key=api_key,
                    model=cfg.get("model", ""),
                    dimensions=cfg.get("dimensions", 1024)
                )
                self.providers.append(provider)

        if not self.providers:
            raise ValueError("没有可用的 Embedding 供应商（请检查 API Key）")

        print(f"✅ Embedding 供应商已初始化: "
              f"{' → '.join(p.name for p in self.providers)}")

    def embed(self, texts: List[str]) -> EmbeddingResult:
        """
        调用 Embedding API（自动 fallback）

        按优先级尝试每个 provider，失败则切换到下一个
        """
        errors = []

        for provider in self.providers:
            if not provider.is_available:
                continue

            try:
                result = provider.embed(texts)
                self.current_provider = provider.name
                self._total_tokens_used += result.token_usage
                return result
            except Exception as e:
                errors.append(f"{provider.name}: {e}")
                continue

        # 所有 provider 都失败了
        error_msg = " | ".join(errors)
        raise RuntimeError(f"所有 Embedding 供应商均失败: {error_msg}")

    def get_stats(self) -> Dict:
        """获取使用统计"""
        return {
            "providers": [
                {
                    "name": p.name,
                    "model": p.model,
                    "available": p.is_available,
                    "healthy": p._healthy
                }
                for p in self.providers
            ],
            "current_provider": self.current_provider,
            "total_tokens_used": self._total_tokens_used
        }


# ========== 向量工具函数 ==========

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    if len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ========== 测试入口 ==========

if __name__ == "__main__":
    print("🧪 多供应商 Embedding 系统测试\n")

    # 从配置文件加载
    config_path = os.path.join(os.path.dirname(__file__), "embedding_config.json")

    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        print("请先创建 embedding_config.json")
        exit(1)

    mp = MultiProviderEmbedding(config_path=config_path)

    # 测试基本嵌入
    print("\n📝 测试 1: 基本嵌入")
    result = mp.embed(["你好世界", "Hello World"])
    print(f"  供应商: {result.provider}")
    print(f"  模型: {result.model}")
    print(f"  维度: {result.dimensions}")
    print(f"  Token 消耗: {result.token_usage}")

    # 测试语义相似度
    print("\n📝 测试 2: 语义相似度")
    result = mp.embed([
        "我喜欢简洁的代码风格",
        "编程规范要求代码简洁明了",
        "今天天气很好适合出去玩"
    ])
    sim_12 = cosine_similarity(result.vectors[0], result.vectors[1])
    sim_13 = cosine_similarity(result.vectors[0], result.vectors[2])
    print(f"  '代码风格' vs '编程规范': {sim_12:.4f} (应该高)")
    print(f"  '代码风格' vs '天气出游': {sim_13:.4f} (应该低)")

    # 状态
    print("\n📊 状态:")
    stats = mp.get_stats()
    for p in stats["providers"]:
        status = "✅" if p["available"] else "❌"
        print(f"  {status} {p['name']} ({p['model']})")
    print(f"  总 Token 消耗: {stats['total_tokens_used']}")
