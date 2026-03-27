# 预计算 Qwen 嵌入使用指南

## 📖 概述

预计算 Qwen 嵌入可以大幅提升训练速度和减少显存占用。

## 🚀 使用方法

### 生成预计算嵌入

```bash
./scripts/1_precompute.sh
```

**时间**: 15分钟
**输出**: `data/embeddings/*.pkl`

### 使用预计算嵌入训练

编辑 `configs/default.yaml`:
```yaml
precomputed:
    use_precomputed: true
```

然后运行:
```bash
./scripts/2_train.sh full
```

## 📊 性能对比

| 指标 | 在线Qwen | 预计算Qwen | 改善 |
|------|----------|------------|------|
| 显存 | 14GB | 5GB | ↓ 64% |
| 速度 | 0.3x | 1.2x | ↑ 4x |

## 🔧 常见问题

**Q: 如何重新生成？**
```bash
rm -rf data/embeddings/
./scripts/1_precompute.sh
```

**Q: 数据更新后怎么办？**
重新运行预计算脚本。
