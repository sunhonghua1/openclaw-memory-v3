# 🧠 OpenClaw Memory Upgrade V3

> 真向量语义搜索 + BM25 混合检索，多供应商 Embedding 自动 Fallback

OpenClaw 增强记忆系统 V3 —— 将 OpenClaw 内置的基础记忆升级为真正的向量语义搜索，支持用不同的词找到含义相同的记忆。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **真向量语义搜索** | 基于 Embedding 的余弦相似度，理解同义词和语义（"编程规范" ↔ "代码风格"） |
| **BM25 混合检索** | 向量搜索 70% + 关键词 30%，兼顾语义和精确匹配 |
| **多供应商 Fallback** | DashScope → Google → Jina AI，任一失败自动切换 |
| **向量缓存** | 本地 JSON 缓存，避免重复 API 调用 |
| **零重型依赖** | 仅使用 Python 标准库（urllib），无需 pip install |
| **分类字典管理** | 记忆按 preference/project/task 等分类存储 |

## 📊 效果演示

```
🔍 查询: '编程规范'
  [0.5255 ✅] 用户喜欢简洁的代码风格，不喜欢过多注释

🔍 查询: '量化策略'
  [1.5771 ✅] 正在开发一个Python量化交易机器人

🔍 查询: '论文进度'
  [1.7069 ✅] ICLR论文截止日期是2026年3月
```

注意：查询词和存储的记忆使用的是**不同的表述**，但语义搜索依然精准命中。

## 🚀 快速安装

### 1. 克隆到 OpenClaw 的 skills 目录

```bash
cd /root/.openclaw/skills/openclaw-memory/
# 备份旧版本
cp openclaw_memory_enhanced.py openclaw_memory_enhanced.py.v2.bak

# 下载新文件
git clone https://github.com/sunhonghua1/openclaw-upgrade.git /tmp/oc-upgrade
cp /tmp/oc-upgrade/embedding_provider.py .
cp /tmp/oc-upgrade/openclaw_memory_enhanced.py .
cp /tmp/oc-upgrade/embedding_config.example.json ./embedding_config.json
```

### 2. 配置 API Key

编辑 `embedding_config.json`，填入你的 API Key：

```bash
nano /root/.openclaw/skills/openclaw-memory/embedding_config.json
```

```json
{
  "primary": "dashscope",
  "providers": {
    "dashscope": {
      "model": "text-embedding-v4",
      "api_key": "你的阿里云 DashScope API Key",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "dimensions": 1024
    },
    "google": {
      "model": "gemini-embedding-001",
      "api_key": "你的 Google Gemini API Key",
      "dimensions": 768
    },
    "jina": {
      "model": "jina-embeddings-v3",
      "api_key": "你的 Jina AI API Key",
      "base_url": "https://api.jina.ai/v1",
      "dimensions": 1024
    }
  }
}
```

> **提示**：不需要三个 provider 都配置，只配一个也能正常工作。推荐至少配置 DashScope 或 Jina（都有免费额度）。

### 3. 获取免费 API Key

| 供应商 | 免费额度 | 获取地址 |
|--------|----------|----------|
| **DashScope** | 100 万 tokens | [阿里云 DashScope](https://dashscope.aliyuncs.com/) |
| **Google Gemini** | 充足 | [Google AI Studio](https://aistudio.google.com/) |
| **Jina AI** | 1000 万 tokens/月 | [Jina AI](https://jina.ai/embeddings/) |

### 4. 测试

```bash
cd /root/.openclaw/skills/openclaw-memory/

# 测试 Embedding 供应商连通性
python3 embedding_provider.py

# 测试完整记忆系统
python3 openclaw_memory_enhanced.py
```

### 5. 重启 OpenClaw

```bash
openclaw gateway restart
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `embedding_provider.py` | 多供应商 Embedding 模块（DashScope/Google/Jina） |
| `openclaw_memory_enhanced.py` | V3 记忆系统核心（混合搜索引擎） |
| `embedding_config.example.json` | 配置模板（需复制为 `embedding_config.json` 并填入 Key） |

## 🏗️ 架构

```
┌──────────────────────────────────────────┐
│   OpenClaw Memory Enhanced V3            │
│   ┌──────────────────────────────────┐   │
│   │   HybridSearchEngine             │   │
│   │   ┌────────────┬───────────────┐ │   │
│   │   │ 向量搜索    │  BM25 关键词   │ │   │
│   │   │ (70%)      │  (30%)        │ │   │
│   │   └──────┬─────┴───────────────┘ │   │
│   │          │                        │   │
│   │   ┌──────▼───────────────────┐   │   │
│   │   │  MultiProviderEmbedding  │   │   │
│   │   │  DashScope → Google → Jina│   │   │
│   │   └──────────────────────────┘   │   │
│   └──────────────────────────────────┘   │
│                                          │
│   ┌────────────┐  ┌──────────────────┐   │
│   │ VectorCache │  │ EnhancedMemoryCore│   │
│   │ (JSON)     │  │ (分类字典管理)    │   │
│   └────────────┘  └──────────────────┘   │
└──────────────────────────────────────────┘
```

## 🔄 与内置记忆的对比

| 能力 | OpenClaw 内置 | V2（升级前） | **V3（本项目）** |
|------|:---:|:---:|:---:|
| 向量语义搜索 | ❌ | ❌ Jaccard 词袋 | ✅ **真 Embedding** |
| "编程规范"匹配"代码风格" | ❌ | ❌ | ✅ |
| BM25 关键词搜索 | ❌ | ✅ | ✅ |
| 多供应商 Fallback | ❌ | ❌ | ✅ |
| 向量缓存 | ❌ | ❌ | ✅ |
| 分类字典管理 | ❌ | ✅ | ✅ |
| 外部依赖 | 无 | 无 | **无** |

## 📜 License

MIT

## 🙏 致谢

- [OpenClaw](https://github.com/nicename-co/openclaw) — AI 助手框架
- [DashScope](https://dashscope.aliyuncs.com/) — 阿里云模型服务
- [Jina AI](https://jina.ai/) — Embedding API
- [Google Gemini](https://ai.google.dev/) — Embedding API
