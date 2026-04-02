# SGSAT-KT 训练脚本

## 核心脚本

```
scripts/
├── 1_precompute.py      # 预计算嵌入
├── 1_precompute.sh      # 预计算快捷启动
├── 2_train.py           # 训练模型
├── 2_train.sh           # 训练快捷启动
├── train.sh             # 一键入口（预计算+训练）
├── utils/               # 工具函数
└── visualization/       # 可视化工具
```

## 使用方法

### 调参
编辑 `../configs/default.yaml` 修改实验配置。

### 预计算
```bash
./scripts/1_precompute.sh
```

### 训练
```bash
./scripts/2_train.sh full    # 完整模型
./scripts/2_train.sh test    # 快速测试
```

## 读取行为说明

- `2_train.py` 会优先按标准 KT 序列格式读取数据。
- v4.2 重建后的 XES 数据已经是标准 `seq_len + q + s` 三行格式。
- 如果检测到历史遗留的非标准文件（例如旧版异常 `xes/train.txt`），训练脚本会自动切换到兼容解析器，而不修改原始 `data/` 文件。
- 兼容解析器只用于排查历史坏数据，不作为当前推荐训练路径。
