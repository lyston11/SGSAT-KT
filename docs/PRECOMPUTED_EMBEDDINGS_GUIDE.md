# TriSG-KT 预计算嵌入使用指南

## 概述

预计算嵌入将 LLM 编码过程从训练循环中分离，一次性生成题目和知识点的语义向量，训练时直接查表使用。

版本演化、协议修订和脚本行为变化，请统一查看 [训练指南](TRAINING.md) 中的“版本记录”章节。

## 当前配置

| 配置项 | 值 |
|--------|-----|
| 模型 | Qwen3-4B (默认) / BERT-base-chinese (备选) |
| 题目嵌入维度 | 2560 (Qwen3) / 768 (BERT) |
| 知识点嵌入维度 | 2560 (Qwen3) / 768 (BERT) |
| 题目数量 | 7618 |
| 知识点数量 | 865 |
| 输出格式 | pickle (.pkl) |

## 使用方法

### 生成预计算嵌入

```bash
./scripts/1_precompute.sh
./scripts/1_precompute.sh algebra05
./scripts/1_precompute.sh algebra05 --device cpu
```

对于 `assist09 / assist17 / statics / doudouyun`，建议先准备文本和图结构：

```bash
python DTransformer/preprocess_data.py --dataset assist09
python DTransformer/preprocess_data.py --dataset assist17
python DTransformer/preprocess_data.py --dataset statics
python DTransformer/preprocess_data.py --dataset doudouyun
```

- 时间: 约15分钟 (Qwen3-4B, 单 GPU)
- 输出: `data/embeddings/{dataset}_question_embeddings.pkl` + `data/embeddings/{dataset}_kc_embeddings.pkl`
- 训练加载时优先按数据集文件名读取，仍兼容旧的全局 `question_embeddings.pkl` / `kc_embeddings.pkl`
- `1_precompute.py` 现在兼容 `question_texts / kc_texts` 中的字符串和字典两种文本格式

### 一键训练（自动检测）

```bash
./scripts/train.sh full
./scripts/train.sh full algebra05
```

脚本会自动检测:
1. pkl 文件是否存在
2. 嵌入维度是否与当前配置的模型 `hidden_size` 匹配
3. 嵌入文件中的 `dataset_name` 是否与当前训练数据集一致
4. 当前数据集中被题目引用的 KC 是否都存在对应预计算嵌入
5. 题目嵌入数量 `q_count` 是否与当前文本数据一致

如果不匹配（如切换了 BERT → Qwen3），会自动重新预计算。

### 手动重新生成

```bash
rm data/embeddings/algebra05_*.pkl
./scripts/1_precompute.sh algebra05
```

## 并行使用建议

- 可以在 `xes` 训练进行时，为 `algebra05` 单独执行预计算，因为两者会写入不同的 embedding 文件
- 同理，也可以为 `assist09 / assist17 / statics / doudouyun` 单独预计算，不会覆盖 `xes` 的 embedding 文件
- 若不想和训练争抢 GPU，建议显式指定 CPU：
  `./scripts/1_precompute.sh algebra05 --device cpu`
- 若希望继续用 GPU 预计算，建议手动约束到未参与训练的设备

## 嵌入如何被使用

```
pkl 文件 (2560维) → PrecomputedEmbeddings (内存查表)
    → PrecomputedEmbeddingLayer
        → Linear(2560→512) → GELU → LN → Linear(512→256) → LN
    → 融合题目 KC 语义: e_q + W_p(e_kc)
    → SSA 选择性语义对齐
    → 门控残差 + LayerNorm
```

### 关键代码路径

| 文件 | 职责 |
|------|------|
| `scripts/1_precompute.py` | 预计算 CLI 入口 |
| `utils/precompute.py` | 预计算模型解析、文本工件读取、KC 文本补全、embedding 生成 |
| `DTransformer/precomputed.py` | `PrecomputedEmbeddings` 加载 pkl |
| `DTransformer/embedding_loader.py` | `PrecomputedEmbeddingLayer` 负责投影与融合层 |
| `DTransformer/model.py` | `LLMGroundingWithPrecomputed` / SSA 对齐融合 ID + LLM + KC 嵌入 |
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

## 文本来源说明

- `xes / algebra05`：使用真实题目文本和知识点描述
- `assist17`：使用原始数据集中可提取的 skill label 生成最小文本
- `assist09 / statics / doudouyun`：当前使用合成文本占位，主要目的是让 `full` 链路和 benchmark 对比实验可以直接运行
