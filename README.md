# SGSAT-KT: Semantic Graph Sparse Attention Knowledge Tracing

基于 SATKT (Sparse Attention Knowledge Tracing) 框架，融合 LLM 语义嵌入和 GNN 知识点先决图的知识追踪模型。

**当前版本**: v1.0 (BERT baseline + Qwen3-4B 预计算嵌入)

---

## 模型原理

### 基座模型: DTransformer (SATKT)

DTransformer 是一种基于稀疏注意力机制的知识追踪模型，核心设计：

1. **距离感知注意力衰减**: 在标准 scaled dot-product attention 基础上，引入可学习的位置距离衰减因子 `gamma`，使模型更关注近期交互
2. **Maxout 稀疏化**: 对注意力权重施加上限约束（最大权重不超过 1.0），实现稀疏注意力
3. **Top-K 选择**: 在注意力计算中，对远距离位置仅保留 Top-K 权重，减少噪声
4. **知识组件发现**: 维护 `n_know=32` 个可学习的知识组件向量，通过交叉注意力将学生状态分解为多个知识维度的掌握程度

### 三项创新扩展

#### Innovation 1: LLM Semantic Grounding (语义嵌入)

将题目文本和知识点文本通过预训练语言模型编码为语义向量，作为题目嵌入的补充信号。

- **预计算模式** (当前默认): 使用 Qwen3-4B (2560维) 或 BERT-base-chinese (768维) 预计算题目/知识点嵌入，训练时直接查表
- **在线模式**: 训练时实时调用 BERT 编码（已冻结，不微调）

融合方式: `q_emb = ID_Embedding(q_id) + Linear(LLM_vec) + W_p(Linear(KC_vec))`

#### Innovation 2: GNN Prerequisite Graph (知识点先决图)

使用 GCN 对知识点先决关系图进行消息传递，捕获知识点间的依赖结构。

- 输入: 先决关系边表 `edge_index` (如 xes 数据集 19506 条边)
- 结构: 2 层 SimpleGCNLayer，带残差连接和 LayerNorm
- 融合方式: `q_emb = q_emb + GNN(kc_ids, edge_index)`

#### Innovation 3: Graph-Enhanced DCF-Sim + Contrastive Learning

对比学习损失，通过序列增强（随机交换相邻位置）和硬负样本（翻转正误标签）构建对比视图。

- 损失: `total = BCE_pred + lambda_cl * CL_loss + reg_loss`

### 已知瓶颈 (v1.0)

当前架构中 LLM 语义信号的有效性受限于：

1. **20:1 压缩瓶颈**: Qwen3-4B 的 2560 维嵌入通过单层 `Linear(2560→128)` 压缩到 128 维，信息损失严重
2. **简单加法融合**: ID embedding、LLM 投影、GNN 嵌入三者直接相加，LLM 信号容易被 ID embedding 淹没
3. **无多层投影**: 仅用单层线性层做维度映射，缺少非线性变换和残差连接

---

## 完整数据流

```
原始数据
├── train.txt / test.txt             → 学生做题序列 (q_id, 正误)
├── xes_question_texts.json          → 题目文本 (content + skill)
├── xes_kc_texts.json                → 知识点文本
├── question_embeddings.pkl          → 预计算嵌入 (6530 × 2560)
├── kc_embeddings.pkl                → 预计算嵌入 (812 × 2560)
└── xes_edge_index.npy               → 知识点先决图 (19506 条边)
    │
    ↓ DataLoader: Batch(q:[B,200], s:[B,200])
    │
Embedding 阶段
    q_id → nn.Embedding → id_emb [B,L,128]
    q_id → pkl查表(2560) → Linear(2560→128) → e_q
    kc_id → pkl查表(2560) → Linear(2560→128) → e_kc → W_p(e_kc)
    kc_id → GCN(edge_index) → prereq_emb
    │
    q_emb = id_emb + e_q + W_p(e_kc) + prereq_emb    [B,L,128]
    s_emb = Embedding(正误) + q_emb                    [B,L,128]
    │
    ↓
Transformer 主干
    Block1: self-attention(s_emb)    → 作答序列自注意力（距离衰减）
    Block2: cross-attention(q→s_emb) → 题目-作答交叉注意力
    │
    ↓ p [B,L,128]
知识组件发现
    know_params [32,128] → expand → Block4 cross-attn
    │
    ↓ z [B,L,4096]  (32知识组件 × 128维)
预测头
    alpha = softmax(know_params @ query)   → 知识组件软分配
    h = alpha @ z                          → 加权知识状态 [B,L,128]
    y = MLP(concat[query, h])              → [B,L,256] → ... → [B,L,1]
    │
    ↓ sigmoid(y) → 预测概率
损失
    weighted_BCE + knowledge_consistency + contrastive_loss + reg_loss
```

### 维度变化总览

| 阶段 | 形状 |
|------|------|
| DataLoader 输出 | `q: (B, 200)`, `s: (B, 200)` |
| 预计算嵌入查表 | `(6530, 2560)` |
| 投影后 | `(B, L, 128)` |
| Transformer 输出 | `(B, L, 128)` |
| 知识组件展开 | `(B*32, L, 128)` |
| 知识组件输出 | `(B, L, 4096)` |
| Readout 后 | `(B, L, 128)` |
| MLP 预测 | `(B, L, 1)` |

---

## 快速开始

### 一键训练（推荐）
```bash
conda activate lyston
./scripts/train.sh full    # 自动预计算 + 训练
```

### 分阶段
```bash
./scripts/1_precompute.sh  # 阶段1: 预计算嵌入
./scripts/2_train.sh full  # 阶段2: 训练
```

### 调参
编辑 `configs/default.yaml`

---

## 项目结构

```
SGSAT-KT/
├── baselines/               # 对比基线 (AKT, DKT, DKVMN, SAKT)
├── configs/
│   └── default.yaml         # 唯一配置来源
├── DTransformer/            # 核心模型代码
│   ├── model.py             # DTransformer 主模型 (注意力、前向传播、损失)
│   ├── data.py              # 数据加载 (KTData, Lines, Batch)
│   ├── embedding_loader.py  # 预计算嵌入加载与投影
│   ├── eval.py              # 评估指标 (AUC, ACC, MAE, RMSE)
│   ├── visualize.py         # 可视化工具
│   ├── preprocess.py        # 难度计算
│   └── preprocess_data.py   # 先决图构建
├── docs/                    # 项目文档
├── scripts/
│   ├── 1_precompute.py      # 预计算 Python 主程序
│   ├── 1_precompute.sh      # 预计算 Shell 脚本
│   ├── 2_train.py           # 训练 Python 主程序
│   ├── 2_train.sh           # 训练 Shell 脚本
│   ├── train.sh             # 一键入口（自动检测嵌入匹配）
│   └── utils/               # 脚本工具模块
├── utils/                   # 通用工具函数
├── data/                    # 数据文件 (只读)
├── pretrained_models/       # 预训练模型权重 (只读)
├── output/                  # 训练输出 (自动生成)
└── logs/                    # 训练日志 (自动生成)
```

---

## 文档

- [训练指南](docs/TRAINING.md)
- [配置指南](docs/CONFIG.md)
- [预计算嵌入指南](docs/PRECOMPUTED_EMBEDDINGS_GUIDE.md)
