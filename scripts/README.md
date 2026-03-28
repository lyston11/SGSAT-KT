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
