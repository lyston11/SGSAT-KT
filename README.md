# SGSAT-KT: Semantic Graph Sparse Attention Knowledge Tracing

---

## 🚀 快速开始

### 新用户（推荐）
```bash
./train.sh full    # 自动预计算 + 训练
```

### 分阶段使用
```bash
# 阶段1: 预计算
./scripts/1_precompute.sh

# 阶段2: 训练
./scripts/2_train.sh full
```

---

## 🔧 调参

编辑 `configs/default.yaml`

---

## 📁 结构

```
scripts/
├── 1_precompute.py      # 预计算
├── 1_precompute.sh      # 预计算Shell
├── 2_train.py           # 训练
└── 2_train.sh           # 训练Shell
```

---

## 📚 文档

- [训练指南](docs/TRAINING.md)
- [配置指南](docs/CONFIG.md)
- [预计算嵌入指南](docs/PRECOMPUTED_EMBEDDINGS_GUIDE.md)
