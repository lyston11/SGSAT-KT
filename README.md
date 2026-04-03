# TriSG-KT: Triple-Sparse Semantic Graph Knowledge Tracing

`TriSG-KT` 是当前论文中使用的正式模型名，对应“`TriSA-Backbone` 原创序列主干 + 语义增强 + 先决图增强 + 多约束协同优化”的完整知识追踪模型。

**当前版本**: v4.3  
详细版本记录见 [训练指南](docs/TRAINING.md)

`v4.3` 聚焦工程化解耦与运行链路模块化，不改变 `TriSA-Backbone + SSA + GNN + MCO` 的方法定义，只重构训练、预计算、数据解析和文档组织边界。

---

## 模型原理

### 基座模型: TriSA-Backbone

`TriSA-Backbone` 是本文当前采用的原创序列主干，实现代码主要位于 `DTransformer/` 目录。其核心设计如下：

1. **距离感知注意力衰减**: 在标准 scaled dot-product attention 基础上，引入可学习的位置距离衰减因子 `gamma`，使模型更关注近期交互
2. **峰值抑噪门**: 对注意力权重施加上限约束，抑制局部异常峰值对状态估计的干扰
3. **关键历程筛选**: 在远距离位置仅保留少量真正关键的历史交互，降低长序列噪声
4. **掌握原型读出**: 维护 `n_know=64` 个可学习的掌握原型，将学生状态分解为多个潜在知识维度
5. **PFE 练习频次编码**: 显式建模同一题目累计出现次数，捕捉“练习次数→掌握程度”的强化/遗忘规律
6. **Embedding 正则化**: Embedding 层 dropout + Cosine Annealing 学习率调度

### 三项创新扩展

#### Innovation 1: SSA Semantic Grounding (语义对齐)

将题目文本和知识点文本通过预训练语言模型编码为语义向量，作为题目嵌入的补充信号。

- **预计算模式** (当前默认): 使用 Qwen3-4B (2560维) 或 BERT-base-chinese (768维) 预计算题目/知识点嵌入，训练时直接查表
- **在线模式**: 训练时实时调用 BERT 编码（已冻结，不微调）

融合方式: `q_emb = LayerNorm(gate * SSA(ID, semantic) + (1-gate) * ID_proj)`
- **SSA 选择性语义对齐**: 题目身份表示主动从外部语义先验中检索与当前预测最相关的语义分量
- **门控残差**: 2 层 gate network 输出标量，防止语义对齐退化为单纯的 ID shortcut

#### Innovation 2: GNN Prerequisite Graph (知识点先决图)

使用 GCN 对知识点先决关系图进行消息传递，捕获知识点间的依赖结构。

- 输入: 先决关系边表 `edge_index` (如 xes 数据集 19506 条边)
- 结构: 2 层 SimpleGCNLayer，带残差连接和 LayerNorm
- 融合方式: `q_emb = q_emb + GNN(kc_ids, edge_index)`

#### Innovation 3: MCO Collaborative Objective（⚠️ DCF-Sim 未集成）

对比学习损失已接入训练主流程。通过序列增强（随机交换相邻位置）和硬负样本（翻转正误标签）构建对比视图。

- 损失: `MCO = BCE_pred + knowledge_consistency + lambda_cl * CL_loss + lambda_contra * InfoNCE + reg_loss`
- `DCFSimGraphEnhanced` 是后处理工具类，用于训练后的用户相似度分析和推荐，不参与模型训练循环
- 配置中 `use_graph_similarity` 仅做标记，不影响训练行为

### 版本演化

详细版本历史、训练协议修订、数据协议修复、脚本行为变化与工程化重构，请统一查看 [训练指南](docs/TRAINING.md) 中的“版本记录”章节。

---

## 完整数据流

```
原始数据
├── train.txt / test.txt             → 学生做题序列 (q_id, 正误)
│                                      若数据集未提供 valid.txt，则训练时从 train.txt
│                                      按固定随机种子切出验证集，test.txt 仅用于最终评估
├── xes_question_texts.json          → 题目文本 (content + skill)
├── xes_kc_texts.json                → 知识点文本
├── xes_question_embeddings.pkl      → 预计算嵌入 (7618 × 2560)
├── xes_kc_embeddings.pkl            → 预计算嵌入 (865 × 2560)
└── xes_edge_index.npy               → 知识点先决图 (19506 条边)
    │
    ↓ DataLoader: Batch(q:[B,200], s:[B,200])
    │
Embedding 阶段 (TriSG-KT / v4.2)
    q_id → Embedding(n_q+1, id_dim=128)      → id_emb [B,L,128]
    q_id → pkl查表(2560) → Linear(2560→512)→GELU→LN→Linear(512→256)→LN → e_q [B,L,256]
    kc_id → pkl查表(2560) → 同样多层投影(独立权重) → e_kc [B,L,256]
    llm_emb = e_q + W_p(e_kc)                          [B,L,256]
    id_proj = Linear(128→256)(id_emb)                   [B,L,256]
    attn_out = SSA(id_proj, llm_emb)                      [B,L,256]
    gate = gate_net(attn_out)                           [B,L,1]
    q_emb = LayerNorm(gate * attn_out + (1-gate) * id_proj) [B,L,256]
    kc_id → GCN(edge_index) → prereq_emb                [B,L,256]
    │
    q_emb = q_emb + prereq_emb                          [B,L,256]
    q_emb = q_emb + repeat_embed(q)                     [B,L,256]
    s_emb = Embedding(正误) + q_emb                     [B,L,256]
    │
    ↓
TriSA-Backbone 序列主干
    Block1: self-attention(s_emb)    → 作答序列自注意力（距离衰减）
    Block2: cross-attention(q→s_emb) → 题目-作答交叉注意力
    │
    ↓ p [B,L,128]
掌握原型读出
    know_params [64,256] → expand → Block4 cross-attn
    │
    ↓ z [B,L,16384]  (64知识组件 × 256维)
预测头
    alpha = softmax(know_params @ query)   → 知识组件软分配
    h = alpha @ z                          → 加权知识状态 [B,L,128]
    y = MLP(concat[query, h])              → [B,L,256] → ... → [B,L,1]
    │
    ↓ sigmoid(y) → 预测概率
MCO 多约束协同目标
    weighted_BCE + knowledge_consistency + lambda_cl * sequence_CL + lambda_contra * embedding_InfoNCE + reg_loss
```

### 维度变化总览

| 阶段 | 形状 |
|------|------|
| DataLoader 输出 | `q: (B, 200)`, `s: (B, 200)` |
| 预计算嵌入查表 | `(7618, 2560)` |
| 多层投影后 | `(B, L, 256)` |
| 门控融合后 | `(B, L, 256)` |
| Transformer 输出 | `(B, L, 256)` |
| 知识组件展开 | `(B*64, L, 256)` |
| 知识组件输出 | `(B, L, 16384)` |
| Readout 后 | `(B, L, 256)` |
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
./scripts/1_precompute.sh                # 按配置中的默认数据集预计算
./scripts/1_precompute.sh algebra05      # 为指定数据集单独预计算
./scripts/2_train.sh full                # 用默认数据集训练
./scripts/2_train.sh full --dataset algebra05
```

多数据集并行使用时，预计算嵌入会按数据集分别保存为 `data/embeddings/{dataset}_question_embeddings.pkl`
和 `data/embeddings/{dataset}_kc_embeddings.pkl`。训练脚本会优先读取对应数据集的文件，
仍兼容旧的全局 `question_embeddings.pkl` / `kc_embeddings.pkl`。

### Benchmark 数据集准备
```bash
python DTransformer/preprocess_data.py --dataset assist09
python DTransformer/preprocess_data.py --dataset assist17
python DTransformer/preprocess_data.py --dataset statics
python DTransformer/preprocess_data.py --dataset doudouyun
```

- `preprocess_data.py` 会按 `data/datasets.toml` 中的 `inputs` 自动解析不同数据集的序列格式
- 会补齐 `data/text_data/{dataset}_question_texts.json`、`data/text_data/{dataset}_kc_texts.json`
- 会生成 `data/processed/{dataset}_edge_index.npy` 和 `data/processed/{dataset}_kc_ids.npy`
- 对 `assist09 / statics / doudouyun` 这类缺少现成文本语料的数据集，当前使用最小合成文本占位来打通 `full` 训练链路，适合 benchmark 对比，不应替代真实文本语义实验

### 调参
编辑 `configs/default.yaml`

### 测试
```bash
make test        # 单元测试 (pytest)
make smoke-test  # 训练冒烟测试 (5轮快速验证)
```

---

## 项目结构

```
<repo_root>/
├── baselines/               # 对比基线 (AKT, DKT, DKVMN, SAKT)
├── configs/
│   └── default.yaml         # 唯一配置来源
├── DTransformer/            # 核心模型代码
│   ├── model.py             # DTransformer 主模型 (主干前向、embedding整合、损失)
│   ├── layers.py            # TriSA-Backbone 注意力层与 attention 算子
│   ├── grounding.py         # 在线 LLM grounding 相关组件
│   ├── graph.py             # GNN 先决图与 DCF-Sim 后处理组件
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
│   ├── project.py           # 项目路径与目录常量
│   ├── experiment.py        # 配置加载、preset 合并、数据集注册表读取
│   ├── data_pipeline.py     # 训练数据源、兼容解析、embedding 校验
│   ├── kt_dataset.py        # KT 序列兼容解析与数据源构建
│   ├── embedding_artifacts.py # 文本/图/embedding 工件加载与校验
│   ├── training.py          # 训练/验证循环、设备初始化、结果落盘
│   ├── preprocessing.py     # 离线预处理复用逻辑
│   ├── precompute.py        # 预计算 embedding 生成与文本工件整理
│   ├── config.py            # 通用配置读写
│   ├── logger.py            # 通用日志工具
│   └── metrics.py           # 通用评估工具
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

## 工程结构说明

为降低脚本级耦合，当前代码库已开始按职责拆分：

- `DTransformer/model.py` 只保留主模型、embedding 聚合和损失逻辑
- `DTransformer/layers.py` 负责注意力层和底层 attention 运算
- `DTransformer/grounding.py` 负责在线 LLM 语义 grounding 组件
- `DTransformer/graph.py` 负责先决图编码和图相似度后处理组件
- `DTransformer/precomputed.py` 负责预计算 embedding 工件存取
- `utils/experiment.py` 统一处理配置/preset 合并
- `utils/kt_dataset.py` 统一处理 KT 数据兼容解析与数据源构建
- `utils/embedding_artifacts.py` 统一处理文本/图/embedding 工件加载和校验
- `utils/data_pipeline.py` 退化为训练侧兼容入口
- `utils/training.py` 统一处理训练/验证循环、设备初始化和结果持久化
- `utils/preprocessing.py` 统一处理序列解析、最小文本生成和先决图构建
- `utils/precompute.py` 统一处理预计算模型解析、文本工件读取和 embedding 生成
- `scripts/2_train.py` 与 `DTransformer/preprocess_data.py` 逐步收敛为 CLI 编排入口
- `scripts/1_precompute.py` 也已收敛为预计算 CLI 编排入口

这轮重构的目标是提升可维护性和模块边界清晰度，不改变当前模型方法和训练语义。
