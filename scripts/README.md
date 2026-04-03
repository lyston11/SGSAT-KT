# TriSG-KT 训练脚本

版本历史、脚本行为变化和工程化重构记录，请统一查看 [训练指南](../docs/TRAINING.md) 中的“版本记录”章节。

## 核心脚本

```
scripts/
├── 1_precompute.py      # 预计算嵌入
├── 1_precompute.sh      # 预计算快捷启动
├── 2_train.py           # 训练模型
├── 2_train.sh           # 训练快捷启动
├── train.sh             # 一键入口（预计算+训练）
├── utils/               # 脚本侧历史工具模块
└── visualization/       # 可视化工具
```

## 使用方法

### 调参
编辑 `../configs/default.yaml` 修改实验配置。

### 预计算
```bash
./scripts/1_precompute.sh
./scripts/1_precompute.sh algebra05
./scripts/1_precompute.sh algebra05 --device cpu
```

### 数据准备
```bash
python DTransformer/preprocess_data.py --dataset assist09
python DTransformer/preprocess_data.py --dataset assist17
python DTransformer/preprocess_data.py --dataset statics
python DTransformer/preprocess_data.py --dataset doudouyun
```

### 训练
```bash
./scripts/2_train.sh full    # 完整模型
./scripts/2_train.sh test    # 快速测试
./scripts/2_train.sh full --dataset algebra05
./scripts/train.sh full algebra05
```

## 读取行为说明

- `2_train.py` 会优先按标准 KT 序列格式读取数据。
- v4.2 重建后的 XES 数据已经是标准 `seq_len + q + s` 三行格式。
- 如果检测到历史遗留的非标准文件（例如旧版异常 `xes/train.txt`），训练脚本会自动切换到兼容解析器，而不修改原始 `data/` 文件。
- 兼容解析器只用于排查历史坏数据，不作为当前推荐训练路径。
- `DTransformer/preprocess_data.py` 会按照 `data/datasets.toml` 中的 `inputs` 自动解析不同数据集的 3/4/6 行 KT 文件布局。
- 对缺少原生题目/知识点文本的数据集，预处理脚本会生成最小合成文本，使 `1_precompute.py` 和 `full` 模式可直接运行。
- 预计算嵌入按数据集分别保存为 `data/embeddings/{dataset}_question_embeddings.pkl`
  和 `data/embeddings/{dataset}_kc_embeddings.pkl`。
- 训练脚本会优先读取对应数据集的嵌入文件，兼容旧全局文件名，因此可以并行准备不同数据集的 embedding。

## 当前工程拆分

当前脚本层已开始复用仓库级公共模块，而不是把所有逻辑都堆在入口脚本中：

- `utils/project.py`：项目根目录和标准路径
- `utils/experiment.py`：`default.yaml` / preset 合并与数据集注册表读取
- `utils/kt_dataset.py`：KT 数据兼容解析、数据源构建、训练内验证集切分
- `utils/embedding_artifacts.py`：文本/图/embedding 工件加载与校验
- `utils/data_pipeline.py`：训练侧兼容入口，聚合上述能力
- `utils/training.py`：训练/验证循环、设备初始化、输出目录与结果持久化
- `utils/preprocessing.py`：离线预处理复用逻辑（序列解析、最小文本生成、图构建）
- `utils/precompute.py`：预计算模型路径解析、文本工件读取、KC 文本补全、embedding 生成

对应地，`DTransformer/` 内部也拆出了：

- `DTransformer/layers.py`
- `DTransformer/grounding.py`
- `DTransformer/graph.py`
- `DTransformer/precomputed.py`

现在：

- `scripts/2_train.py` 更接近纯编排入口
- `DTransformer/preprocess_data.py` 更接近纯 CLI 入口
- `scripts/1_precompute.py` 也更接近纯 CLI 入口

目标是让入口脚本只负责编排，而不是同时承担路径、配置、数据和模型组件定义。
