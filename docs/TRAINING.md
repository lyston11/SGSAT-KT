# SGSAT-KT 训练指南

## 模型版本

**v3.1** — 基座模型优化: n_know=64 + cosine annealing + 重复次数嵌入

| 组件 | 版本/配置 |
|------|----------|
| 基座模型 | DTransformer (SATKT) |
| LLM (默认) | Qwen3-4B, 预计算嵌入, hidden_size=2560 |
| LLM (备选) | BERT-base-chinese, 在线/预计算, hidden_size=768 |
| GNN | 2层 SimpleGCNLayer |
| d_model | 256 |
| id_dim | 128 (ID embedding 固定) |
| llm_proj_dim | 256 (LLM 投影目标) |
| llm_inter_dim | 512 (LLM 中间投影) |
| id_dropout_rate | 0.15 |
| lambda_contra | 0.3 |
| contrast_temperature | 0.07 |
| 知识组件数 | 64 (v3.1: 32→64) |
| 重复嵌入 | max_repeats=20 (v3.1 新增) |
| 学习率调度 | Cosine Annealing (v3.1 新增) |
| 注意力头数 | 8 |

## 两段式训练

```bash
# 阶段1: 预计算（一次性，约15分钟）
./scripts/1_precompute.sh

# 阶段2: 训练（可重复）
./scripts/2_train.sh full
```

一键入口会自动检测嵌入维度是否匹配当前配置的模型，不匹配时自动重新预计算：
```bash
./scripts/train.sh full
```

## 模型架构详解

### Embedding 阶段

v3.0 采用门控融合机制动态平衡 ID 和 LLM 信号：

```
# 多层投影（保留 v2.0 改进）
llm_vec = pkl查表(2560) → Linear(2560→512)→GELU→LN→Linear(512→256)→LN

# 门控融合（v3.0 新增）
gate = sigmoid(Linear(256)(llm_vec))        # 门控值 [0,1]
gated_llm = gate * llm_vec                   # 门控 LLM 信号
id_proj = Linear(128→256)(ID_Embedding)      # ID 投影到同维度
q_emb = LayerNorm(gated_llm + id_proj)       # 同维度融合

# GNN 先决图（保留）
q_emb = q_emb + GNN(kc_ids, edge_index)
```

- **ID Embedding**: `nn.Embedding(n_questions+1, 128)` — 可学习的题目 ID 查表
- **ID Dropout** (v3.0): 训练时 p=0.15 将 ID 置零，迫使模型依赖 LLM
- **LLM 门控**: `sigmoid(W_g · llm_vec) * llm_vec` — 动态调节 LLM 信号贡献
- **KC 融合门**: `W_p = nn.Linear(256, 256, bias=False)` — 知识点嵌入融合权重
- **GNN**: `SimpleGCNLayer × 2` — 先决图消息传递，带残差 LayerNorm

作答嵌入: `s_emb = nn.Embedding(2, 256)(s) + q_emb`

### Transformer 主干 (n_layers=2)

```
Block1: self-attention(s_emb)           — 学生作答序列自注意力
Block2: cross-attention(Q=q_emb, V=s_emb) — 题目-作答交叉注意力
```

每个 Block 包含:
- Multi-Head Attention (8头, d_k=16)
- 距离感知衰减 (可学习 gamma)
- Maxout 稀疏化 (权重上限 1.0)
- Top-K 选择 (远距离位置)
- LayerNorm + Dropout

### 知识组件发现

```
know_params [64, 256]  →  expand  →  Block4 cross-attn(know→hidden)  →  z [B,L,16384]
```

### 预测头

```
alpha = softmax(know_params @ query)  → 知识组件软分配
h = alpha @ z                         → 加权知识状态
y = MLP(concat[query, h])             → 256 → 256 → 128 → 1
```

MLP 结构: `Linear(512→256) → GELU → Dropout → Linear(256→128) → GELU → Dropout → Linear(128→1)`

### 损失函数

**标准模式** (baseline/test):
```
loss = weighted_BCE + 0.05 * knowledge_consistency + reg_loss
```

**完整模式** (full/prod, `cl_loss=True`):
```
loss = weighted_BCE + lambda_cl * sequence_CL + lambda_contra * embedding_InfoNCE + reg_loss
```

- `weighted_BCE`: 错题权重 1.2 的二元交叉熵
- `knowledge_consistency`: 余弦相似度惩罚，鼓励知识组件表示多样性
- `sequence_CL`: 序列增强 (随机交换) + 硬负样本 (翻转标签) 序列级对比学习，温度 0.05
- `embedding_InfoNCE` (v3.0 新增): LLM 投影空间 in-batch 对比损失，同 KC 题目靠近，不同 KC 题目远离，温度 0.07
- `reg_loss`: `p_diff^2 * 1e-3`（仅使用 pid 时）

## 可用模式

| 模式 | 命令 | 说明 | LLM | GNN | CL | 轮数 |
|------|------|------|-----|-----|-----|------|
| `test` | `./scripts/2_train.sh test` | 快速验证 | OFF | OFF | OFF | 5 |
| `baseline` | `./scripts/2_train.sh baseline` | 基线 | OFF | OFF | OFF | 100 |
| `full` | `./scripts/2_train.sh full` | 完整模型 | ON | ON | ON | 30 |
| `prod` | `./scripts/2_train.sh prod` | 生产环境 | ON | ON | OFF | 200 |
| `sakt` | `./scripts/2_train.sh sakt` | SAKT 基线 | OFF | OFF | OFF | 100 |
| `akt` | `./scripts/2_train.sh akt` | AKT 基线 | OFF | OFF | OFF | 100 |
| `dkt` | `./scripts/2_train.sh dkt` | DKT 基线 | OFF | OFF | OFF | 100 |
| `dkvmn` | `./scripts/2_train.sh dkvmn` | DKVMN 基线 | OFF | OFF | OFF | 100 |

## 调参

编辑 `configs/default.yaml`：

```yaml
training:
    batch_size: 16
    n_epochs: 30
    learning_rate: 0.001

llm:
    use_llm: true
    pretrained_model: "pretrained_models/qwen3-4b"  # 或 bert-base-chinese

gnn:
    use_gnn: true
```

## 分支状态诊断

训练启动后会打印分支激活状态，可快速核验：

- `use_precomputed: True` — 预计算嵌入已接入前向传播
- `use_online_text: True` — 在线文本编码分支已启用
- `use_gnn: True` + `gnn_edges: <num>` — 先决图已接入
- `use_cl_loss: True` — 对比学习损失已启用

## 版本历史

### v3.1 (当前)
- n_know 扩容: 32→64，匹配 xes 数据集 812 个知识点的表达需求
- Cosine Annealing 学习率调度: 替代固定 lr，后期更稳定
- Embedding dropout: q_emb/s_emb 在送入 Transformer 前施加 dropout
- 重复次数嵌入: 通过同一题目累计出现次数建模遗忘/强化信号

### v3.0
- 门控融合: `sigmoid(W_g · llm) * llm + Linear(id)` 替代 concat+project
- 辅助 InfoNCE 对比损失: 强制 LLM 投影空间结构化
- ID Dropout (p=0.15): 训练时随机置零 ID，迫使模型依赖 LLM

### v2.0
- 多层漏斗投影: `2560 → 512 → 256` 两层 GELU+LN 投影
- 拼接融合: ID(128) + LLM(256) = 384 → Linear(384, 256)
- d_model 从 128 提升到 256

**v2.0 问题**: concat 融合允许 `W_llm` 退化为零，LLM 信号未参与训练。

### v1.0
- 单层 Linear(2560→128) 压缩
- 简单加法融合
- 20:1 信息瓶颈，LLM 信号被 ID 淹没

## 常见问题

**Q: 预计算嵌入不存在？**
```bash
./scripts/1_precompute.sh
```

**Q: 切换了模型但嵌入维度不匹配？**
```bash
./scripts/train.sh full  # 自动检测并重新预计算
```

**Q: 显存不足？**

减小 `batch_size` 或关闭 AMP，参考 [配置指南](CONFIG.md)
