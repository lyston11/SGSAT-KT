# TriSG-KT 训练指南

## 模型版本

**v4.3** — 基于 v4.2 方法定义的工程化解耦、多数据集流程整理与文档收口版

| 组件 | 版本/配置 |
|------|----------|
| 基座模型 | TriSA-Backbone（实现位于 `DTransformer/`） |
| LLM (默认) | Qwen3-4B, 预计算嵌入, hidden_size=2560 |
| LLM (备选) | BERT-base-chinese, 在线/预计算, hidden_size=768 |
| GNN | 2层 SimpleGCNLayer |
| d_model | 256 |
| id_dim | 128 (ID embedding 固定) |
| llm_proj_dim | 256 (LLM 投影目标) |
| llm_inter_dim | 512 (LLM 中间投影) |
| id_dropout_rate | 0.15 |
| lambda_contra | 0.3 |
| contrast_temperature | 0.07 |
| 知识组件数 | 64 (v3.1: 32→64) |
| 重复嵌入 | max_repeats=20 (v3.1 新增) |
| 学习率调度 | Cosine Annealing (v3.1 新增) |
| 注意力头数 | 8 |
| Cross-Attn 头数 | 4 (v4.0 新增, v4.1 延续) |

## 两段式训练

```bash
# 阶段1: 预计算（一次性，约15分钟）
./scripts/1_precompute.sh
./scripts/1_precompute.sh algebra05 --device cpu

# 阶段2: 训练（可重复）
./scripts/2_train.sh full
./scripts/2_train.sh full --dataset algebra05
```

一键入口会自动检测嵌入维度、数据集元信息和 KC 覆盖是否匹配当前配置，不匹配时自动重新预计算：
```bash
./scripts/train.sh full
./scripts/train.sh full algebra05
```

- 预计算嵌入现在按数据集保存为 `data/embeddings/{dataset}_question_embeddings.pkl`
  和 `data/embeddings/{dataset}_kc_embeddings.pkl`
- 训练时会优先读取对应数据集的嵌入文件，仍兼容历史全局文件名
- 因此可以在 `xes` 训练进行时，为 `algebra05` 单独预计算，不会覆盖 `xes` 的 embedding 工件

## 工程结构

训练入口正在逐步收敛为“编排层”：

- `scripts/2_train.py` 负责训练编排和结果落盘
- `utils/experiment.py` 负责配置/preset 合并
- `utils/kt_dataset.py` 负责数据源构建、兼容解析和训练内验证集切分
- `utils/embedding_artifacts.py` 负责文本/图/embedding 工件加载与校验
- `utils/data_pipeline.py` 负责训练侧兼容聚合
- `utils/training.py` 负责训练/验证循环、运行设备初始化、输出工件持久化
- `utils/preprocessing.py` 负责离线数据预处理复用逻辑
- `utils/precompute.py` 负责预计算模型解析、文本工件加载和 embedding 生成
- `DTransformer/model.py` 只保留主模型与损失逻辑
- `DTransformer/layers.py` / `grounding.py` / `graph.py` / `precomputed.py` 承载可复用模型组件
- `DTransformer/preprocess_data.py` 只保留 benchmark 数据预处理 CLI 编排

这轮拆分的目标是提升可维护性，而不是改变训练行为。

## Benchmark 数据集准备

在 `assist09 / assist17 / statics / doudouyun` 上运行 `full` 之前，先补齐文本和图结构产物：

```bash
python DTransformer/preprocess_data.py --dataset assist09
python DTransformer/preprocess_data.py --dataset assist17
python DTransformer/preprocess_data.py --dataset statics
python DTransformer/preprocess_data.py --dataset doudouyun
```

- 该脚本会读取 `data/datasets.toml` 中的 `inputs` 配置，自动适配 3 行、4 行或 6 行布局的 KT 文件
- 会生成 `data/text_data/{dataset}_question_texts.json`、`data/text_data/{dataset}_kc_texts.json`
- 会生成 `data/processed/{dataset}_edge_index.npy`、`data/processed/{dataset}_kc_ids.npy`
- `assist17` 会尽量利用原始 skill label 生成最小文本
- `assist09 / statics / doudouyun` 目前使用合成文本占位，用来打通 LLM 预计算和 `full` 模式训练
- 这类合成文本适合 benchmark 对比和链路复现，不应替代 `xes / algebra05` 上的真实语义实验

## 模型架构详解

### Embedding 阶段

v4.2 延续 SSA 选择性语义对齐机制，ID embedding 作为身份触发端主动检索外部语义先验：

```
# 多层投影（保留 v2.0 改进）
llm_vec = pkl查表(2560) → Linear(2560→512)→GELU→LN→Linear(512→256)→LN

# SSA 语义对齐（v4.0 引入，v4.2 延续）
id_proj = Linear(128→256)(ID_Embedding)      # ID 投影到同维度
attn_out = SSA(id_proj, llm_vec)             # 身份触发语义检索
gate = gate_net(attn_out)                     # 2层gate→标量 [0,1]
fused = gate * attn_out + (1-gate) * id_proj  # 门控残差
q_emb = LayerNorm(fused)                      # 同维度融合

# GNN 先决图（保留）
q_emb = q_emb + GNN(kc_ids, edge_index)
```

- **ID Embedding**: `nn.Embedding(n_questions+1, 128)` — 可学习的题目 ID 查表
- **ID Dropout** (v3.0): 训练时 p=0.15 将 ID 置零，迫使模型依赖 LLM
- **SSA 选择性语义对齐** (v4.0+): ID embedding 作为身份触发端，主动检索 task-specific 语义分量
- **门控残差** (v4.0+): 2 层 gate network (Linear→ReLU→Linear→Sigmoid) → 标量，防止 attention 坍塌
- **KC 融合门**: `W_p = nn.Linear(256, 256, bias=False)` — 知识点嵌入融合权重
- **GNN**: `SimpleGCNLayer × 2` — 先决图消息传递，带残差 LayerNorm

作答嵌入: `s_emb = nn.Embedding(2, 256)(s) + q_emb`

### TriSA-Backbone 序列主干 (n_layers=2)

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
know_params [64, 256]  →  expand  →  Block4 cross-attn(know→hidden)  →  z [B,L,16384]
```

### 预测头

```
alpha = softmax(know_params @ query)  → 知识组件软分配
h = alpha @ z                         → 加权知识状态
y = MLP(concat[query, h])             → 256 → 256 → 128 → 1
```

MLP 结构: `Linear(512→256) → GELU → Dropout → Linear(256→128) → GELU → Dropout → Linear(128→1)`

### MCO 多约束协同目标

**标准模式** (baseline/test):
```
loss = weighted_BCE + 0.05 * knowledge_consistency + reg_loss
```

**完整模式** (full/prod, `cl_loss=True`):
```
loss = weighted_BCE + 0.05 * knowledge_consistency + lambda_cl * sequence_CL + lambda_contra * embedding_InfoNCE + reg_loss
```

- `weighted_BCE`: 错题权重 1.2 的二元交叉熵
- `knowledge_consistency`: 余弦相似度惩罚，鼓励掌握原型表示多样性
- `sequence_CL`: 序列增强 (随机交换) + 硬负样本 (翻转标签) 序列级对比学习，温度 0.05
- `embedding_InfoNCE` (v3.0 新增): LLM 投影空间 in-batch 对比损失，同 KC 题目靠近，不同 KC 题目远离，温度 0.07
- `reg_loss`: `p_diff^2 * 1e-3`（仅使用 pid 时）

## 数据划分策略

- 如果数据集提供 `valid`，训练过程使用该验证集做模型选择。
- 如果数据集未提供 `valid`，训练脚本会按 `training.validation_ratio` 和 `training.validation_seed` 从 `train.txt` 中确定性切出验证集。
- `test.txt` 仅在训练结束后对最佳验证模型做一次最终评估，不参与模型选择和早停。

## 数据读取兼容性

- `2_train.py` 默认按标准 KT 文件格式读取数据。
- v4.2 重建后的 XES 数据已经恢复为标准三行格式。
- 兼容解析器仍然保留，用于排查历史异常文件；若检测到旧的非标准文件布局，训练脚本会自动启用兼容解析器而不修改原始 `data/` 文件。
- `DTransformer/preprocess_data.py` 则负责离线准备 benchmark 数据集的文本和图结构，并统一按 `data/datasets.toml` 解析不同字段布局。

## 输出产物

- `best_model.pt`: 最佳验证模型参数
- `metrics_history.json`: 每个 epoch 的训练损失与验证指标
- `summary.json`: 最佳验证结果、最终测试结果与 split 信息
- `split_info.json`: 训练/验证划分策略与参数

## 可用模式

| 模式 | 命令 | 说明 | LLM | GNN | CL | 轮数 |
|------|------|------|-----|-----|-----|------|
| `test` | `./scripts/2_train.sh test` | 快速验证 | OFF | OFF | OFF | 5 |
| `baseline` | `./scripts/2_train.sh baseline` | 基线 | OFF | OFF | OFF | 100 |
| `full` | `./scripts/2_train.sh full` | 完整模型 | ON | ON | ON | 30 |
| `prod` | `./scripts/2_train.sh prod` | 生产环境 | ON | ON | OFF | 200 |
| `sakt` | `./scripts/2_train.sh sakt` | SAKT 基线 | OFF | OFF | OFF | 100 |
| `akt` | `./scripts/2_train.sh akt` | AKT 基线 | OFF | OFF | OFF | 100 |
| `dkt` | `./scripts/2_train.sh dkt` | DKT 基线 | OFF | OFF | OFF | 100 |
| `dkvmn` | `./scripts/2_train.sh dkvmn` | DKVMN 基线 | OFF | OFF | OFF | 100 |

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

## 版本记录

本节是项目唯一正式版本记录入口。  
训练协议、数据协议、脚本行为变化、工程化重构和版本演化，统一维护在本文件中。

### v4.3（当前）

版本定位：  
`v4.3` 不改变 `TriSG-KT` 的模型数学定义、主干结构和联合优化目标，重点对工程实现、运行链路、模块边界和文档入口做系统收敛。

#### 设计目标

- 降低训练脚本、预计算脚本、模型组件之间的耦合
- 让 `scripts/` 入口只负责编排，把公共逻辑沉到 `utils/` 与 `DTransformer/` 子模块
- 支持多数据集预处理、预计算和训练流程更稳定地并行运行
- 统一版本历史入口，避免文档职责重叠

#### 工程化与模块化重构

- 将训练/验证循环、设备初始化、AMP 逻辑、最佳模型保存、指标历史落盘、训练总结落盘，从训练脚本中抽离到 `utils/training.py`
- 将 `scripts/2_train.py` 收敛为训练编排入口，主要保留模式解析、配置加载、数据准备、模型构建和训练调度
- 将 KT 序列标准解析、异常历史格式兼容解析、训练内验证集切分逻辑抽离到 `utils/kt_dataset.py`
- 将文本工件、先决图工件、预计算 embedding 工件的读取与校验逻辑抽离到 `utils/embedding_artifacts.py`
- `utils/data_pipeline.py` 退化为训练侧兼容聚合入口，只保留 batch 级辅助函数和对外稳定接口
- 将预计算模型路径解析、文本工件读取、KC 文本回填、embedding 生成逻辑抽离到 `utils/precompute.py`
- 将 `scripts/1_precompute.py` 收敛为预计算 CLI 编排入口
- 将 `PrecomputedEmbeddings` 从 `DTransformer/embedding_loader.py` 中拆出到 `DTransformer/precomputed.py`
- `DTransformer/embedding_loader.py` 只保留预计算语义融合层实现
- 将原先集中在 `DTransformer/model.py` 中的组件进一步分离到：
  - `DTransformer/layers.py`
  - `DTransformer/grounding.py`
  - `DTransformer/graph.py`
  - `DTransformer/precomputed.py`
- `DTransformer/model.py` 保留主模型、主损失链路和关键前向逻辑
- 上述重构不改变模型数学定义、主干设计、损失目标和训练方法

#### 文档同步

- `README.md` 更新为当前模块边界
- `scripts/README.md` 同步脚本侧结构重构
- `docs/CONFIG.md` 回收重复历史说明，只保留当前配置背景
- `docs/PRECOMPUTED_EMBEDDINGS_GUIDE.md` 同步预计算模块边界

#### 验证与清理

- 验证通过：
  - `python scripts/1_precompute.py --help`
  - `python scripts/2_train.py --help`
  - `python DTransformer/preprocess_data.py --help`
  - 相关模块导入 smoke test
  - 相关文件 `py_compile`
- 清理完成：
  - `__pycache__`
  - `/tmp/*.pyc`
  - 临时兼容软链接

#### 影响范围

- 不改变已有 `TriSG-KT` 方法定义、配置语义和训练目标
- 主要影响代码组织方式、运行入口职责划分和文档检索路径
- 为后续继续接入 benchmark 数据集、做消融和论文整理提供更稳定的工程底座

### v4.2

版本定位：  
`v4.2` 是基于 `v4.0` SSA 语义对齐架构的训练/评估链路与 XES 数据协议修订版。

核心特征：

- 方法主链路不变：`TriSA-Backbone + SSA + GNN + MCO`
- 重点工作转向数据协议、文本协议、图结构协议与预计算 embedding 协议统一

#### 设计目标

- 修复 XES 历史数据文件格式不稳定的问题
- 修复题目文本、知识点文本、知识点图、配置和 embedding 工件之间的编号不一致问题
- 保证训练流程建立在统一、稳定、可重跑的数据协议之上

#### XES 数据协议重建

- 修复 `process_xes.py` 的项目根路径计算错误，重新对齐原始 `xes_math.csv`
- 重新生成标准三行 KT 文件：
  - `train.txt`
  - `valid.txt`
  - `test.txt`
- 采用学生级别稳定划分
- 避免旧版混合格式文件继续参与正式训练

#### 文本与知识点编号修复

- `xes_question_texts.json` 中的 `skill` 从历史稀疏/原始标识改为 dense `kc_id`
- 同时保留 `raw_skill` 字段用于追溯
- 让题目文本、先决图、模型配置、embedding 工件共享同一套 KC 编号体系

#### 数据规模更新

- XES 当前协议更新为：
  - `n_questions = 7618`
  - `n_kc = 865`
- 对应 embedding 工件规模同步更新：
  - 题目 embedding：7618 条
  - 知识点 embedding：865 条

#### 训练入口联动修复

- `train.sh` 新增 `q_count` 校验
- 旧的 `6530 / 828` embedding 工件会被自动识别为失效
- 发现旧工件时自动触发重算，避免静默复用错误 embedding

#### 兼容策略

- 保留 `2_train.py` 中的异常文件兼容解析器
- 兼容解析器只作为历史问题排查兜底
- 当前推荐训练路径仍是标准协议重建后的数据文件

#### 多数据集 embedding 隔离

- 预计算结果按数据集独立保存
- 支持在 `xes` 训练进行时并行为 `algebra05` 等数据集准备 embedding
- 避免不同数据集共享旧全局 embedding 文件而相互污染

#### 实际效果

- 修复训练过程中由异常历史文件触发的序列长度类错误
- 重建 XES 的统一题目/KC 编号体系
- 为后续 benchmark 与论文实验复现提供稳定底座

### v4.1

版本定位：  
`v4.1` 是在 `v4.0` SSA 架构稳定后，对训练协议、验证职责、预计算工件校验和实验追溯能力的系统修复版。

#### 设计目标

- 修复训练/验证/测试职责边界不清问题
- 修复预计算 embedding 被静默误用的问题
- 修复训练目标和文档定义不完全一致的问题
- 提升实验结果追溯能力

#### 训练/评估协议修复

- 无 `valid` 数据集时，从 `train.txt` 按固定随机种子切出验证集
- `test.txt` 只在训练结束后用于最终评估
- 训练过程中不再使用测试集做模型选择

#### 联合优化目标对齐

- 在 `cl_loss=True` 时补回 `knowledge_consistency`
- 保证代码中的训练目标与文档定义一致

#### KC 语义链路修复

- 只要启用 `use_llm`，训练与验证阶段都会一致构造 `kc_ids`
- 不再错误依赖 `use_gnn` 才能启用 KC 语义链路

#### 预计算工件校验增强

- embedding 工件增加 `dataset_name` 元信息
- 训练前自动检查：
  - 数据集是否匹配
  - 题目 embedding 数量是否覆盖当前数据
  - 当前数据引用的 KC 是否都具有 embedding

#### 梯度累积修复

- 按真实反向传播步数统计梯度累积
- 兼容序列切块和多视图训练场景

#### 结果追溯补强

- 新增：
  - `metrics_history.json`
  - `summary.json`
  - `split_info.json`
- 训练结果不再只依赖日志文本

#### 实际影响

- 训练/验证/测试职责边界更加清晰
- embedding 误用风险显著下降
- 实验记录更适合论文回填与复现

### v4.0

版本定位：  
`v4.0` 是 `TriSG-KT` SSA 语义对齐架构正式确立的版本。

#### 设计背景

`v3.0` 的门控融合仍然属于逐位置强度调节：

- `gate * llm_emb + id_proj`

该结构存在关键局限：

- 冻结预计算得到的 `llm_emb` 是 task-agnostic 的
- 简单门控只能控制强弱，不能主动选择当前预测真正相关的语义分量
- 模型仍可能主要依赖 ID shortcut

#### SSA 选择性语义对齐

- 使用 ID embedding 作为身份触发端
- 主动从外部语义先验中检索 task-specific 语义分量
- 不再只对固定语义向量做线性门控

#### 门控残差

- 使用两层 gate network：
  - `Linear -> ReLU -> Linear -> Sigmoid`
- gate 输出标量
- 融合形式：
  - `LayerNorm(gate * attn_out + (1-gate) * id_proj)`
- 防止 attention 退化到纯 ID shortcut

#### Causal Mask

- 保持 KT 自回归约束
- 不允许语义对齐路径访问未来位置

#### 训练链路与配置修复

- 修复 `cross_attn_heads` 没有正确从 yaml 传入模型的问题
- 修复 `freeze_bert` 配置传递问题
- 修复在线 LLM fallback 时 list/tensor 类型冲突
- 修复退化到 ID 路径时的维度不匹配（`128 -> 256`）
- 修复 shell 脚本在 `test / baseline` 等非 LLM 模式下仍错误要求 embedding 工件的问题
- 修复预计算 embedding 检查缺少 `use_llm` 守卫的问题

#### 配置与依赖清理

- 删除多项无效配置项
- 清理历史死代码结构
- 对齐 `pyproject.toml` 与 `requirements.txt`
- 修复依赖拼写与版本对齐问题

#### 工具链与测试补强

- `scripts/1_precompute.py` 开始支持数据集参数
- `DCFSimGraphEnhanced` 明确标注为后处理工具类
- `make test` 统一为 pytest 单测入口
- 新增 `make smoke-test` 训练冒烟测试
- 明确 `DataParallel` 对当前 `get_loss/predict` 结构实际无效，改为单 GPU 规范行为

#### 其他缺陷修复

- 修复 DKT baseline sigmoid 双重应用问题
- 修复 `s_emb` 构造顺序，确保包含完整特征
- 修复 `knowledge_consistency_loss` 对 padding 位置处理不严谨问题
- 修复 `kc_ids` 加载逻辑不再错误依赖 `use_gnn=true`

#### 实际影响

- `TriSG-KT` 的核心 SSA 语义对齐机制在该版本正式成形
- 主训练链路、配置链路、辅助脚本链路基本打通

### v3.1

版本定位：  
`v3.1` 是对 `TriSA-Backbone` 表达能力和训练稳定性的增强版本。

#### 主要更新

- `n_know` 从 `32` 扩容到 `64`
- 引入 Cosine Annealing 学习率调度
- 在 embedding 层加入 dropout 正则化
- 新增重复次数嵌入 `max_repeats=20`

#### 设计意图

- 更匹配 XES 等知识点规模更大的数据集
- 提升后期训练稳定性
- 更显式建模重复练习、强化与遗忘现象

### v3.0

版本定位：  
`v3.0` 是对早期语义融合失败问题的第一次实质性修正版本。

#### 主要更新

- 用门控融合替代简单 concat+project
- 引入 embedding 空间辅助 InfoNCE 对比损失
- 引入 `ID Dropout (p=0.15)`

#### 设计意图

- 防止 LLM 信号在训练中被完全忽略
- 强制语义投影空间具备结构化区分能力
- 迫使模型在训练时更多利用语义而不是纯题目 ID

### v2.0

版本定位：  
`v2.0` 是对早期单层压缩方案的第一次升级，尝试通过更宽的投影链路保留更多语义信息。

#### 主要更新

- 引入多层漏斗式投影：
  - `2560 -> 512 -> 256`
- 使用 `GELU + LayerNorm`
- 使用拼接融合：
  - `ID(128) + LLM(256) = 384 -> Linear(384, 256)`
- `d_model` 从 `128` 提升到 `256`

#### 主要问题

- concat 融合允许融合层权重对 LLM 分支退化
- 模型仍可能忽略 LLM 信号
- 因而没有获得预期质量提升

### v1.0

版本定位：  
`v1.0` 是最早的语义增强尝试版本。

#### 主要更新

- 使用单层 `Linear(2560 -> 128)` 压缩语义 embedding
- 与 ID 表示做简单加法融合

#### 主要问题

- 存在严重信息瓶颈
- 高维语义信息在入口处大量损失
- 最终 LLM 信号容易被 ID 表示淹没

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
