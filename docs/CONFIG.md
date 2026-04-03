# TriSG-KT 配置指南

## 📝 重要提示

**只需修改一个配置文件：`configs/default.yaml`**

不需要修改任何代码！

## 工程化拆分

当前项目已开始按职责拆分公共能力，避免训练入口脚本和单个模型文件持续膨胀：

- `utils/project.py`：统一项目路径
- `utils/experiment.py`：统一 `default.yaml` / preset 合并与数据集注册表读取
- `utils/data_pipeline.py`：统一数据源构建、异常序列兼容解析、预计算 embedding 校验
- `DTransformer/layers.py`：注意力层与底层 attention
- `DTransformer/grounding.py`：在线语义 grounding 组件
- `DTransformer/graph.py`：GNN 先决图和图相似度后处理组件

当前原则是：不改变模型方法和训练语义，只降低代码耦合度。

---

## 🚀 快速开始

### 选择训练模式

```bash
./scripts/train.sh test      # 快速测试（5轮）
./scripts/train.sh dtransformer  # 官方 DTransformer 基线
./scripts/train.sh full      # 完整模型（LLM+GNN）← 推荐
./scripts/train.sh full algebra05  # 指定数据集
./scripts/train.sh prod      # 生产环境（200轮）
```

### 修改配置

编辑 `configs/default.yaml`，所有参数都在这里！

---

## 🎮 GPU 配置

### 单GPU训练

```yaml
gpu:
  device_ids: [0]  # 只使用GPU 0
```

### 候选 GPU 列表

当前训练脚本会在 `device_ids` 中自动选择最空闲的一张 GPU，而不是启用 `DataParallel`：

```yaml
gpu:
  device_ids: [0, 1]  # 从 GPU 0 和 1 中自动选择空闲显存更多的一张
```

### 特定GPU

```yaml
gpu:
  device_ids: [1, 2]  # 只使用GPU 1和2
```

## 🤖 LLM 模型配置

### 修改模型路径

编辑 `configs/default.yaml` 中的 `llm.pretrained_model`：

```yaml
llm:
  pretrained_model: "pretrained_models/qwen3-4b"
```

### 使用 HuggingFace 模型

```yaml
llm:
  pretrained_model: "pretrained_models/qwen3-4b"
  # 或者
  pretrained_model: "pretrained_models/bert-base-chinese"
```

## 🏋️ 训练配置

### 预设模式（推荐）

在 `configs/default.yaml` 的 `presets` 部分有以下预设：

```yaml
presets:
  test:      # 快速验证（5轮，无LLM/GNN）
  dtransformer:  # 官方 DTransformer 基线（100轮，无LLM/GNN）
  full:      # 完整模型（30轮，LLM+GNN）
  prod:      # 生产环境（200轮，最佳性能）
  sakt:      # SAKT 基线（100轮）
  akt:       # AKT 基线（100轮）
  dkt:       # DKT 基线（100轮）
  dkvmn:     # DKVMN 基线（100轮）
```

使用方式：
```bash
conda activate lyston
./scripts/train.sh full   # 自动应用 full 预设的配置
./scripts/train.sh full algebra05
```

### 自定义参数

修改 `configs/default.yaml`：

```yaml
training:
  batch_size: 16        # 批大小（双GPU建议16）
  n_epochs: 30          # 训练轮数
  learning_rate: 0.001  # 学习率
  device: "cuda"        # 设备 (cuda 或 cpu)
  validation_ratio: 0.1 # 无 valid 时从 train 切分验证集比例
  validation_seed: 42   # 无 valid 时的固定随机种子
```

### 功能开关

```yaml
llm:
  use_llm: true          # 是否使用LLM嵌入

gnn:
  use_gnn: true          # 是否使用GNN

recommendation:
  use_graph_similarity: false  # DCFSimGraphEnhanced 标记（后处理工具，不影响训练）
  cl_loss: true              # 是否使用对比损失
```

### 当前关键参数

```yaml
model:
  id_dropout_rate: 0.15    # ID embedding dropout 概率（迫使模型依赖 LLM）
  n_know: 64               # 知识组件数量（v3.1: 32→64）

llm:
  lambda_contra: 0.3          # 辅助嵌入对比损失权重
  contrast_temperature: 0.07  # InfoNCE 温度
```

### 当前配置说明

- `n_know = 64`：当前主干使用 64 个知识组件
- `id_dropout_rate = 0.15`：训练时随机削弱 ID shortcut，迫使模型利用语义分支
- `lambda_contra = 0.3`：embedding 级 InfoNCE 约束权重
- `contrast_temperature = 0.07`：InfoNCE 温度参数
- `cross_attn_heads = 4`：SSA 选择性语义对齐头数
- XES 当前配置基于重建后的干净协议：
  - `n_questions = 7618`
  - `n_kc = 865`

配置项的历史演化、版本修订背景和各版本修复范围，请统一查看 [训练指南](TRAINING.md) 中的“版本记录”章节。

### Benchmark 数据集文本/图准备

运行 `assist09 / assist17 / statics / doudouyun` 的 `full` 模式前，先执行：

```bash
python DTransformer/preprocess_data.py --dataset assist09
python DTransformer/preprocess_data.py --dataset assist17
python DTransformer/preprocess_data.py --dataset statics
python DTransformer/preprocess_data.py --dataset doudouyun
```

- 脚本会根据 `data/datasets.toml` 的 `inputs` 自动解析序列布局
- 会补齐 `data/text_data/{dataset}_question_texts.json` 和 `data/text_data/{dataset}_kc_texts.json`
- 会生成 `data/processed/{dataset}_edge_index.npy` 和 `data/processed/{dataset}_kc_ids.npy`
- `assist17` 会尽量利用原始 skill label；其余缺少文本的数据集会生成最小合成文本占位
- 因此这些 benchmark 数据集现在也能直接走 `1_precompute.sh {dataset}` 和 `2_train.sh full --dataset {dataset}`

## 🚀 使用方法

### 方法1: 一键训练（推荐）

```bash
# 使用默认配置（full模式）
./scripts/train.sh full
./scripts/train.sh full algebra05

# 使用其他模式
./scripts/train.sh test      # 快速测试
./scripts/train.sh dtransformer  # 官方 DTransformer 基线
./scripts/train.sh prod      # 生产环境
```

### 方法2: 分步训练

```bash
# 步骤1: 预计算嵌入（仅第一次需要）
./scripts/1_precompute.sh
./scripts/1_precompute.sh algebra05 --device cpu

# 步骤2: 训练模型
./scripts/2_train.sh full
./scripts/2_train.sh full --dataset algebra05
```

## 📥 下载Qwen模型

### 选项1: 使用 HuggingFace CLI

```bash
cd pretrained_models/

# 安装工具
pip install huggingface-hub

# 下载模型（推荐 Qwen2.5-1.5B，约3GB）
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir Qwen2.5-1.5B-Instruct

# 然后修改 configs/default.yaml:
# llm.pretrained_model: "pretrained_models/Qwen2.5-1.5B-Instruct"
```

### 选项2: 从 ModelScope 下载（国内更快）

```bash
pip install modelscope

cd pretrained_models/

python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',
                  cache_dir='.',
                  revision='master')
"

# 然后修改 configs/default.yaml:
# llm.pretrained_model: "Qwen/Qwen2.5-1.5B-Instruct"
```

### 选项3: 浏览器下载

1. 访问: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
2. 点击 "Download" 下载 zip 文件
3. 上传到服务器并解压到 `pretrained_models/` 目录

## 🔍 检查配置

### 查看当前配置

```bash
python -c "
from scripts.utils.config_loader import ConfigLoader
loader = ConfigLoader()
config = loader.load_yaml('default.yaml')
import yaml
print(yaml.dump(config, allow_unicode=True))
"
```

### 测试模型加载

```bash
python -c "
from scripts.utils.config_loader import ConfigLoader
from transformers import AutoModel

loader = ConfigLoader()
config = loader.load_yaml('default.yaml')
model_path = config['llm']['pretrained_model']

print(f'加载模型: {model_path}')
model = AutoModel.from_pretrained(model_path)
print(f'✅ 模型加载成功!')
print(f'隐藏层大小: {model.config.hidden_size}')
"
```

## 🎯 常见问题

### Q1: 显存不足怎么办？

减小批大小：
```yaml
# configs/default.yaml
presets:
  full:
    training:
      batch_size: 8   # 从16改为8
```

### Q2: 如何查看有多少GPU？

```bash
nvidia-smi
```

### Q3: 如何只使用部分GPU？

修改 `configs/default.yaml`:
```yaml
gpu:
  device_ids: [0, 1]  # 只使用GPU 0和1
```

### Q4: 显存不足怎么办？

1. 减小批大小：
```yaml
training:
  batch_size: 32  # 从64改为32
```

2. 使用更少的GPU：
```yaml
gpu:
  device_ids: [0]  # 只用1块GPU
```

### Q5: 模型路径在哪里配置？

在 `configs/default.yaml` 第18行：
```yaml
llm:
  pretrained_model: "你的模型路径"
```

## 📊 训练输出位置

训练结果保存在 `output/` 目录：

```
output/
└── runs/
    └── xes/
        └── full/
            └── 2026-04-03/
                └── 155344_qwen3-4b/
                    ├── artifacts/
                    │   └── best_model.pt
                    ├── metrics/
                    │   ├── metrics_history.json
                    │   └── summary.json
                    └── meta/
                        ├── config.json
                        ├── split_info.json
                        └── run_info.json
```

- `artifacts/` 保存模型权重
- `metrics/` 保存结构化结果
- `meta/` 保存配置、划分信息和运行状态
- 训练日志统一在 `logs/` 目录，不再写入 `output/`

## 🔄 切换预设配置

项目内置了几个预设配置，在 `configs/default.yaml` 的 `presets` 中：

- **test**: 快速测试（5 epochs，无LLM/GNN）
- **dtransformer**: 官方 DTransformer 基线（100 epochs，无LLM/GNN）
- **full**: 完整模型（100 epochs，LLM+GNN）
- **prod**: 生产环境（200 epochs，更大批大小）

```bash
./scripts/train.sh test      # 测试
./scripts/train.sh dtransformer  # 官方 DTransformer 基线
./scripts/train.sh full      # 完整
./scripts/train.sh prod      # 生产
./scripts/train.sh sakt      # SAKT 基线
./scripts/train.sh akt       # AKT 基线
./scripts/train.sh dkt       # DKT 基线
./scripts/train.sh dkvmn     # DKVMN 基线
```

## 📞 需要帮助？

检查日志文件：`logs/training_*.log`
或者查看错误堆栈信息。
