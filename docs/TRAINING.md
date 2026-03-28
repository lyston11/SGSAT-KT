# SGSAT-KT 训练指南

## 模型版本

**v1.0** — 基于 SATKT 框架，扩展 LLM 语义嵌入 + GNN 先决图 + 对比学习

| 组件 | 版本/配置 |
|------|----------|
| 基座模型 | DTransformer (SATKT) |
| LLM (默认) | Qwen3-4B, 预计算嵌入, hidden_size=2560 |
| LLM (备选) | BERT-base-chinese, 在线/预计算, hidden_size=768 |
| GNN | 2层 SimpleGCNLayer |
| d_model | 128 |
| 知识组件数 | 32 |
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

题目嵌入由三路信号相加得到：

```
q_emb = ID_Embedding(q_id) + proj_q(LLM_vec) + W_p(proj_kc(KC_vec)) + GNN(kc_ids)
```

- **ID Embedding**: `nn.Embedding(n_questions+1, 128)` — 可学习的题目 ID 查表
- **LLM 投影**: `nn.Linear(hidden_size, 128)` — 预训练模型嵌入的线性投影（可学习）
- **KC 融合门**: `W_p = nn.Linear(128, 128, bias=False)` — 知识点嵌入融合权重
- **GNN**: `SimpleGCNLayer × 2` — 先决图消息传递，带残差 LayerNorm

作答嵌入: `s_emb = nn.Embedding(2, 128)(s) + q_emb`

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
know_params [32, 128]  →  expand  →  Block4 cross-attn(know→hidden)  →  z [B,L,4096]
```

### 预测头

```
alpha = softmax(know_params @ query)  → 知识组件软分配
h = alpha @ z                         → 加权知识状态
y = MLP(concat[query, h])             → 256 → 256 → 128 → 1
```

MLP 结构: `Linear(256→256) → GELU → Dropout → Linear(256→128) → GELU → Dropout → Linear(128→1)`

### 损失函数

**标准模式** (baseline/test):
```
loss = weighted_BCE + 0.05 * knowledge_consistency + reg_loss
```

**完整模式** (full/prod, `cl_loss=True`):
```
loss = weighted_BCE + lambda_cl * contrastive_loss + reg_loss
```

- `weighted_BCE`: 错题权重 1.2 的二元交叉熵
- `knowledge_consistency`: 余弦相似度惩罚，鼓励知识组件表示多样性
- `contrastive_loss`: 序列增强 (随机交换) + 硬负样本 (翻转标签) 对比学习，温度 0.05
- `reg_loss`: `p_diff^2 * 1e-3`（仅使用 pid 时）

## 可用模式

| 模式 | 命令 | 说明 | LLM | GNN | CL | 轮数 |
|------|------|------|-----|-----|-----|------|
| `test` | `./scripts/2_train.sh test` | 快速验证 | OFF | OFF | OFF | 5 |
| `baseline` | `./scripts/2_train.sh baseline` | 基线 | OFF | OFF | OFF | 100 |
| `full` | `./scripts/2_train.sh full` | 完整模型 | ON | ON | ON | 30 |
| `prod` | `./scripts/2_train.sh prod` | 生产环境 | ON | ON | OFF | 200 |

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

## v1.0 已知瓶颈

1. **20:1 压缩瓶颈**: Qwen3-4B 2560维 → 单层 Linear → 128维，信息损失严重
2. **简单加法融合**: ID/LLM/GNN 三路信号直接相加，LLM 信号容易被 ID embedding 淹没
3. **无多层投影**: 缺少非线性变换和残差连接

> 这些瓶颈是下一版本迭代的核心优化方向。

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
