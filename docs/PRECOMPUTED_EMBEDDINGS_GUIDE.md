# 预计算嵌入使用指南

## 概述

预计算嵌入将 LLM 编码过程从训练循环中分离，一次性生成题目和知识点的语义向量，训练时直接查表使用。

## 当前配置

| 配置项 | 值 |
|--------|-----|
| 模型 | Qwen3-4B (默认) / BERT-base-chinese (备选) |
| 题目嵌入维度 | 2560 (Qwen3) / 768 (BERT) |
| 知识点嵌入维度 | 2560 (Qwen3) / 768 (BERT) |
| 题目数量 | 6530 |
| 知识点数量 | 812 |
| 输出格式 | pickle (.pkl) |

## 使用方法

### 生成预计算嵌入

```bash
./scripts/1_precompute.sh
```

- 时间: 约15分钟 (Qwen3-4B, 单 GPU)
- 输出: `data/embeddings/question_embeddings.pkl` + `kc_embeddings.pkl`

### 一键训练（自动检测）

```bash
./scripts/train.sh full
```

脚本会自动检测:
1. pkl 文件是否存在
2. 嵌入维度是否与当前配置的模型 `hidden_size` 匹配
3. 嵌入文件中的 `dataset_name` 是否与当前训练数据集一致
4. 当前数据集中被题目引用的 KC 是否都存在对应预计算嵌入

如果不匹配（如切换了 BERT → Qwen3），会自动重新预计算。

### 手动重新生成

```bash
rm data/embeddings/*.pkl
./scripts/1_precompute.sh
```

## 嵌入如何被使用

```
pkl 文件 (2560维) → PrecomputedEmbeddings (内存查表)
    → PrecomputedEmbeddingLayer
        → Linear(2560→512) → GELU → LN → Linear(512→256) → LN
    → 融合题目 KC 语义: e_q + W_p(e_kc)
    → ID Query attend LLM Key/Value (Cross-Attention)
    → 门控残差 + LayerNorm
```

### 关键代码路径

| 文件 | 职责 |
|------|------|
| `scripts/1_precompute.py` | 生成嵌入 pkl 文件 |
| `DTransformer/embedding_loader.py` | `PrecomputedEmbeddings` 加载 pkl, `PrecomputedEmbeddingLayer` 投影 |
| `DTransformer/model.py` | `LLMGroundingWithPrecomputed` 融合 ID + LLM + KC 嵌入 |
| `scripts/2_train.py` | 训练时加载并传入模型 |

## 性能对比

| 指标 | 在线编码 | 预计算嵌入 |
|------|----------|------------|
| 显存占用 | ~14GB | ~5GB |
| 训练速度 | 0.3x | 1.2x |
| 模型质量 | 相同 | 相同 |

## 数据格式

pkl 文件结构:
```python
{
    "question_ids": ["0", "1", "2", ...],      # 题目/知识点 ID 列表
    "embeddings": np.ndarray,                    # shape: (N, hidden_size)
    "hidden_size": 2560,                         # 嵌入维度
    "model_path": "pretrained_models/qwen3-4b",  # 使用的模型路径
    "dataset_name": "xes"                        # 当前嵌入所属数据集
}
```
