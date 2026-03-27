# SGSAT-KT 训练指南

## 两段式训练

项目默认就是两段式：预计算（一次性）+ 训练（可重复）。

```bash
# 阶段1: 预计算（一次性）
./scripts/1_precompute.sh

# 阶段2: 训练（重复）
./scripts/2_train.sh full
```

## 调参

编辑 `configs/default.yaml`：

```yaml
training:
    dataset: "xes"
    n_epochs: 150
    batch_size: 96

llm:
    use_llm: true

gnn:
    use_gnn: true
```

## 可用模式

| 模式 | 命令 | 说明 | 时间 |
|------|------|------|------|
| `test` | `./scripts/2_train.sh test` | 快速测试 | 2分钟 |
| `baseline` | `./scripts/2_train.sh baseline` | 基线模型 | 30分钟 |
| `full` | `./scripts/2_train.sh full` | 完整模型 | 1小时 |

## Full 模式说明

- 当 `precomputed.use_precomputed: true` 时，训练会加载 `data/embeddings/*.pkl` 并进入预计算嵌入分支。
- 当 `gnn.use_gnn: true` 时，训练会尝试加载 `data/processed/<dataset>_edge_index.npy`（或 `data/embeddings/processed/<dataset>_edge_index.npy`）。
- 当 `recommendation.cl_loss: true` 时，训练会调用对比学习损失分支。

训练启动后会打印 `分支激活状态`，可快速核验：

- `use_precomputed: True` 表示预计算嵌入已真实接入前向。
- `use_online_text: True` 表示在线文本编码分支已启用。
- `use_gnn: True` 且出现 `gnn_edges: <num>` 表示先决图已接入。
- `use_cl_loss: True` 表示训练将调用对比损失分支。

## 常见问题

**Q: 预计算嵌入不存在？**
```bash
./scripts/1_precompute.sh
```

**Q: 一条命令自动走两段式？**
```bash
./train.sh full
```

