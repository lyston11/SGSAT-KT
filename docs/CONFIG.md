# SGSAT-KT 配置指南

## 📝 重要提示

**只需修改一个配置文件：`configs/default.yaml`**

不需要修改任何代码！

---

## 🚀 快速开始

### 选择训练模式

```bash
./scripts/train.sh test      # 快速测试（5轮）
./scripts/train.sh baseline  # 基线模型
./scripts/train.sh full      # 完整模型（LLM+GNN）← 推荐
./scripts/train.sh prod      # 生产环境（200轮）
```

### 修改配置

编辑 `configs/default.yaml`，所有参数都在这里！

---

## 🎮 GPU 配置

### 1️⃣ 单GPU训练（默认）

编辑 `configs/default.yaml`：

```yaml
gpu:
  device_ids: null  # null表示自动使用所有可用GPU
  primary_gpu: 0
  use_data_parallel: false
```

### 2️⃣ 多GPU训练

指定使用哪些GPU（例如使用GPU 0, 1, 2, 3）：

```yaml
gpu:
  device_ids: [0, 1, 2, 3]  # 使用4块GPU
  primary_gpu: 0            # 主GPU
  use_data_parallel: true   # 启用DataParallel
```

### 3️⃣ 特定GPU

只使用GPU 1和2：

```yaml
gpu:
  device_ids: [1, 2]
  primary_gpu: 1
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

在 `configs/default.yaml` 的 `presets` 部分有4个预设：

```yaml
presets:
  test:      # 快速验证（5轮，无LLM/GNN）
  baseline:  # 基线模型（100轮，无LLM/GNN）
  full:      # 完整模型（100轮，LLM+GNN）
  prod:      # 生产环境（200轮，最佳性能）
  sakt:      # SAKT 基线（100轮）
  akt:       # AKT 基线（100轮）
  dkt:       # DKT 基线（100轮）
  dkvmn:     # DKVMN 基线（100轮）
```

使用方式：
```bash
./scripts/train.sh full   # 自动应用 full 预设的配置
```

### 自定义参数

修改 `configs/default.yaml`：

```yaml
training:
  batch_size: 16        # 批大小（双GPU建议16）
  n_epochs: 100         # 训练轮数
  learning_rate: 0.001  # 学习率
  device: "cuda"        # 设备 (cuda 或 cpu)
```

### 功能开关

```yaml
llm:
  use_llm: true          # 是否使用LLM嵌入

gnn:
  use_gnn: true          # 是否使用GNN

recommendation:
  use_graph_similarity: true  # 是否使用图相似度
  cl_loss: true              # 是否使用对比损失
```

### v3.0 新增参数

```yaml
model:
  id_dropout_rate: 0.15    # ID embedding dropout 概率（迫使模型依赖 LLM）
  n_know: 64               # 知识组件数量（v3.1: 32→64）

llm:
  lambda_contra: 0.3          # 辅助嵌入对比损失权重
  contrast_temperature: 0.07  # InfoNCE 温度
```

### v3.1 基座模型优化

- `n_know: 32→64`: 知识组件扩容，匹配 xes 数据集 ~812 个知识点
- Cosine Annealing 学习率调度: 自动配置，无需手动设置
- 重复次数嵌入: `max_repeats=20`，建模题目重复出现次数（遗忘/强化信号）
- Embedding dropout: 训练时对 q_emb/s_emb 施加 dropout 正则化

## 🚀 使用方法

### 方法1: 一键训练（推荐）

```bash
# 使用默认配置（full模式）
./scripts/train.sh full

# 使用其他模式
./scripts/train.sh test      # 快速测试
./scripts/train.sh baseline  # 基线模型
./scripts/train.sh prod      # 生产环境
```

### 方法2: 分步训练

```bash
# 步骤1: 预计算嵌入（仅第一次需要）
./scripts/1_precompute.sh

# 步骤2: 训练模型
./scripts/2_train.sh full
```

## 📥 下载Qwen模型

### 选项1: 使用 HuggingFace CLI

```bash
cd /home1/LT/SGSAT-KT/pretrained_models/

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

cd /home1/LT/SGSAT-KT/pretrained_models/

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
├── full_xes_20250321_120000/
│   ├── config.json          # 配置文件
│   ├── best_model.pt        # 最佳模型
│   └── training.log         # 训练日志
```

## 🔄 切换预设配置

项目内置了几个预设配置，在 `configs/default.yaml` 的 `presets` 中：

- **test**: 快速测试（5 epochs，无LLM/GNN）
- **baseline**: 基线模型（100 epochs，无LLM/GNN）
- **full**: 完整模型（100 epochs，LLM+GNN）
- **prod**: 生产环境（200 epochs，更大批大小）

```bash
./scripts/train.sh test      # 测试
./scripts/train.sh baseline  # 基线
./scripts/train.sh full      # 完整
./scripts/train.sh prod      # 生产
./scripts/train.sh sakt      # SAKT 基线
./scripts/train.sh akt       # AKT 基线
./scripts/train.sh dkt       # DKT 基线
./scripts/train.sh dkvmn     # DKVMN 基线
```

## 📞 需要帮助？

检查日志文件：`output/*/training.log`
或者查看错误堆栈信息。
