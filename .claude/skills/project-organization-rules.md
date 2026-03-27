# 项目文件组织规范 (Project Organization Rules)

**适用场景：** 所有软件项目，特别是机器学习/深度学习项目
**目的：** 保持项目结构清晰，文件归档有序，便于维护和协作

---

## 📁 标准目录结构

### 必需目录

```
project_root/
├── docs/              # 所有文档、总结、实验结果
├── logs/              # 所有训练/运行日志
├── scripts/           # 所有可执行脚本
├── output/            # 模型输出、检查点、结果
├── data/              # 数据文件
├── tests/             # 测试脚本
└── utils/             # 工具函数
```

### 根目录保留文件

**仅保留这些文件在根目录：**
- `README.md` - 项目说明
- `requirements.txt` - Python依赖
- `pyproject.toml` - Poetry配置（如使用）
- `Makefile` - 快捷命令（如使用）
- `CHANGELOG.md` - 变更日志
- `run_*.sh` - 主启动脚本（最多1-2个）

**❌ 不要在根目录放置：**
- 临时实验脚本
- 日志文件
- 中间结果文件
- 配置文件（应放在configs/）
- 测试文件（应放在tests/）

---

## 📝 文件分类规则

### 1. 文档文件 (docs/)

**位置：** `docs/`

**包含：**
- 实验结果总结：`EXPERIMENT_RESULTS.md`
- 训练状态：`TRAINING_STATUS.md`
- 基线对比：`baseline_summary.txt`
- 项目结构：`PROJECT_STRUCTURE.md`
- API文档、设计文档等

**命名规范：**
- 使用清晰描述性名称
- Markdown格式优先（.md）
- 纯文本次之（.txt）

### 2. 日志文件 (logs/)

**位置：** `logs/`

**命名格式：**
```
{app_name}_{dataset}_{model}_{timestamp}.log
```

**示例：**
```
logs/train_xes_baseline_20260307_143000.log
logs/train_xes_llm_20260307_150000.log
logs/train_xes_gnn_20260307_160000.log
```

**❌ 不要在根目录或临时位置创建日志**

**脚本中的日志输出规范：**
```bash
# ✅ 正确：使用绝对路径
nohup python script.py > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ❌ 错误：会在当前目录创建
nohup python script.py > train.log 2>&1 &
```

### 3. 脚本文件 (scripts/)

**位置：** `scripts/`

**包含：**
- 训练脚本：`train.py`, `train_llm.py`
- 测试脚本：`test_*.py`
- 工具脚本：`preprocess_data.py`
- 监控脚本：`monitor_*.sh`

**❌ 不要在根目录放置：**
- `test_*.py` → 应放在 `tests/`
- `monitor_*.sh` → 应放在 `scripts/`
- `preprocess_*.py` → 应放在 `scripts/` 或 `utils/`

### 4. 输出文件 (output/)

**位置：** `output/`

**子目录结构：**
```
output/
├── {dataset}_{model}/
│   ├── model-001-0.8240.pt
│   ├── config.json
│   └── checkpoints/
```

**示例：**
```
output/
├── xes_baseline/
├── xes_llm/
├── xes_gnn/
└── xes_llm_gnn/
```

### 5. 配置文件 (configs/)

**位置：** `configs/`

**包含：**
- 模型配置：`model_config.yaml`
- 训练配置：`train_config.json`
- 实验配置：`experiments.toml`

---

## 🔧 脚本编写规范

### 日志输出

**所有训练/运行脚本必须：**

1. **指定完整日志路径**
```python
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{LOG_DIR}/train_{timestamp}.log"

# 使用
subprocess.run(f"python train.py > {log_file} 2>&1 &", shell=True)
```

2. **提供日志查看命令**
```bash
echo "✓ 训练已启动"
echo "📄 日志文件: {log_file}"
echo "💡 查看实时日志: tail -f {log_file}"
```

### 输出文件

**模型保存必须：**
1. 创建专门的输出目录
2. 使用描述性目录名
3. 保存配置信息

```python
import os
import json

output_dir = "output/xes_llm"
os.makedirs(output_dir, exist_ok=True)

# 保存配置
config_path = os.path.join(output_dir, "config.json")
json.dump(config, open(config_path, "w"))

# 保存模型
model_path = os.path.join(output_dir, f"model-{epoch:03d}-{auc:.4f}.pt")
torch.save(model.state_dict(), model_path)
```

---

## 📋 文件整理检查清单

### 创建新项目时

- [ ] 创建标准目录：docs/, logs/, scripts/, output/, tests/, utils/
- [ ] 设置日志输出路径到 `logs/`
- [ ] 设置模型输出路径到 `output/`
- [ ] 创建 `docs/PROJECT_STRUCTURE.md`
- [ ] 将文档文件放到 `docs/`

### 创建新脚本时

- [ ] 确定脚本类型（训练/测试/工具）
- [ ] 放到正确目录：
  - 训练脚本 → `scripts/`
  - 测试脚本 → `tests/`
  - 工具函数 → `utils/`
- [ ] 如果输出日志，使用 `logs/{name}_{timestamp}.log`
- [ ] 如果保存模型，使用 `output/{experiment_name}/`

### 训练完成后

- [ ] 检查日志是否在 `logs/`
- [ ] 检查模型是否在 `output/`
- [ ] 在 `docs/` 创建实验总结
- [ ] 更新 `docs/EXPERIMENT_RESULTS.md`
- [ ] 清理临时文件

---

## 🛠️ 快捷命令

### 项目初始化

```bash
mkdir -p docs logs scripts output tests utils
touch docs/PROJECT_STRUCTURE.md
touch docs/EXPERIMENT_RESULTS.md
```

### 日志查看

```bash
# 最新日志
tail -f logs/*.log

# 特定训练日志
tail -f logs/train_xes_*.log

# 搜索错误
grep -i "error\|exception" logs/*.log
```

### 文件整理

```bash
# 整理根目录的文档
mv *.md docs/ 2>/dev/null
mv *.txt docs/ 2>/dev/null

# 整理根目录的脚本
mv test_*.py tests/ 2>/dev/null
mv *.sh scripts/ 2>/dev/null
mv preprocess_*.py scripts/ 2>/dev/null

# 整理根目录的日志
mv *.log logs/ 2>/dev/null
```

---

## ⚠️ 常见错误及纠正

### 错误1：日志文件散落在各处

**症状：**
```bash
$ find . -name "*.log"
./train.log
./experiment1.log
./logs/train_xes.log
./output/eval.log
```

**纠正：**
```bash
# 移动所有日志到logs/
find . -name "*.log" -not -path "./logs/*" -exec mv {} logs/ \;
```

### 错误2：文档文件混杂

**症状：**
```bash
$ ls *.md *.txt
README.md
EXPERIMENT_RESULTS.md
baseline_summary.txt
PROJECT_STRUCTURE.md
```

**纠正：**
```bash
# 保留README.md在根目录，其他移到docs/
mv EXPERIMENT_RESULTS.md docs/
mv PROJECT_STRUCTURE.md docs/
mv baseline_summary.txt docs/
```

### 错误3：临时脚本未清理

**症状：**
```bash
$ ls *.py *.sh
test_integration.py
run_experiment.sh
quick_test.py
```

**纠正：**
```bash
# 测试脚本
mv test_*.py tests/

# 一次性实验脚本 → 移到docs/或删除
rm quick_test.py

# 可复用脚本 → 移到scripts/
mv run_experiment.sh scripts/
```

---

## 📊 项目健康度评分

### 评分标准

**满分：100分**

- [ ] 根目录只有标准文件（-10分）
- [ ] 所有日志在logs/（-20分）
- [ ] 所有文档在docs/（-20分）
- [ ] 所有脚本在正确目录（-20分）
- [ ] 所有输出在output/（-20分）
- [ ] 有PROJECT_STRUCTURE.md（-10分）

**当前项目状态：**
- ✅ 根目录整洁
- ✅ 日志归档到logs/
- ✅ 文档归档到docs/
- ✅ 脚本归档到scripts/
- ✅ 输出归档到output/
- ✅ 有项目结构说明

**总分：100分** 🎉

---

## 🎯 实施原则

### 1. 预防优于治理
- 项目开始时就建立正确结构
- 每个新文件都要问："这个文件应该放在哪里？"

### 2. 一致性优先
- 所有项目使用相同的目录结构
- 所有脚本使用相同的命名规范
- 所有日志使用相同的时间戳格式

### 3. 文档同步更新
- 每次实验后更新docs/
- 每次结构调整后更新PROJECT_STRUCTURE.md
- 保持README.md与实际结构一致

### 4. 定期清理
- 每周检查一次根目录
- 每月清理旧日志
- 每个阶段结束后整理文档

---

**版本：** v1.0
**创建时间：** 2026-03-07
**适用项目：** 所有机器学习/软件项目
**维护者：** Claude Code Assistant
