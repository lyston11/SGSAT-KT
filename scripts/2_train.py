#!/usr/bin/env python3
"""
阶段2: 训练模型
使用预计算嵌入进行高效训练
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, project_root)

# 导入配置加载器
from scripts.utils.config_loader import ConfigLoader

from DTransformer.data import KTData
from DTransformer.eval import Evaluator
from DTransformer.model import DTransformer
from DTransformer.embedding_loader import PrecomputedEmbeddings


def merge_dicts(base, update):
    """递归合并字典"""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config, parent_key='', sep='_'):
    """将嵌套的配置字典扁平化"""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # 递归扁平化子字典
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
        # 同时保留原始嵌套结构的键
        items.append((k, v))

    # 添加常用别名
    flat_dict = dict(items)
    # n_epochs -> epochs
    if 'training_n_epochs' in flat_dict:
        flat_dict['epochs'] = flat_dict['training_n_epochs']
        flat_dict['n_epochs'] = flat_dict['training_n_epochs']
    if 'training_batch_size' in flat_dict:
        flat_dict['batch_size'] = flat_dict['training_batch_size']
    if 'training_learning_rate' in flat_dict:
        flat_dict['learning_rate'] = flat_dict['training_learning_rate']
    if 'training_device' in flat_dict:
        flat_dict['device'] = flat_dict['training_device']
    if 'llm_use_llm' in flat_dict:
        flat_dict['use_llm'] = flat_dict['llm_use_llm']
    if 'gnn_use_gnn' in flat_dict:
        flat_dict['use_gnn'] = flat_dict['gnn_use_gnn']
    if 'llm_pretrained_model' in flat_dict:
        flat_dict['pretrained_model'] = flat_dict['llm_pretrained_model']
    if 'precomputed_use_precomputed' in flat_dict:
        flat_dict['use_precomputed'] = flat_dict['precomputed_use_precomputed']

    return flat_dict


def load_text_data(dataset_name, data_dir):
    """加载题目文本数据"""
    text_data_dir = os.path.join(data_dir, "text_data")
    q_text_path = os.path.join(text_data_dir, f"{dataset_name}_question_texts.json")

    if os.path.exists(q_text_path):
        with open(q_text_path, 'r', encoding='utf-8') as f:
            q_texts = json.load(f)
        print(f"✓ 加载了 {len(q_texts)} 个题目文本")
        return q_texts
    else:
        return None


def load_q_to_kc_mapping(dataset_name, data_dir):
    """加载q_id到kc_id的映射"""
    text_data_path = os.path.join(data_dir, "text_data", f"{dataset_name}_question_texts.json")

    if not os.path.exists(text_data_path):
        return None

    with open(text_data_path, 'r', encoding='utf-8') as f:
        q_texts = json.load(f)

    q_to_kc = {}
    for q_id_str, q_data in q_texts.items():
        q_id = int(q_id_str)
        skill = q_data.get("skill", "-1")
        try:
            kc_id = int(skill)
            if kc_id >= 0:
                q_to_kc[q_id] = kc_id
        except:
            pass

    print(f"✓ 创建了 {len(q_to_kc)} 个q_id->kc_id映射")
    return q_to_kc


def prepare_bert_inputs(texts, q_ids, tokenizer, max_length=128):
    """准备BERT输入"""
    if texts is None:
        return None

    batch_size, seq_len = q_ids.shape
    all_texts = []

    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            q_id = str(q_ids[batch_idx, seq_idx].item())
            if q_id in texts and q_id != "0":
                text = texts[q_id].get("content", "")
                if "Unknown" not in text and text.strip():
                    all_texts.append(text)
                else:
                    all_texts.append("[PAD]")
            else:
                all_texts.append("[PAD]")

    if not all_texts or all(t == "[PAD]" for t in all_texts):
        return None

    from transformers import AutoTokenizer
    encoded = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }


def add_kc_ids_to_batch(q_batch, q_to_kc_mapping, n_kc=None):
    """添加kc_ids到批次数据

    Args:
        q_batch: 题目ID批次
        q_to_kc_mapping: q_id到kc_id的映射
        n_kc: 知识点数量，用于边界检查
    """
    # 处理 list 和 tensor 两种情况
    if isinstance(q_batch, list):
        # 如果是 list，对每个元素处理
        return [add_kc_ids_to_batch(q, q_to_kc_mapping, n_kc) for q in q_batch]

    batch_size, seq_len = q_batch.shape
    kc_ids = torch.zeros_like(q_batch)

    for b in range(batch_size):
        for s in range(seq_len):
            q_id = q_batch[b, s].item()
            if q_id in q_to_kc_mapping:
                kc_id = q_to_kc_mapping[q_id]
                # 边界检查：确保 kc_id < n_kc
                if n_kc is not None and kc_id >= n_kc:
                    kc_ids[b, s] = 0  # 超出范围的设为 0
                else:
                    kc_ids[b, s] = kc_id
            else:
                kc_ids[b, s] = 0

    return kc_ids


def load_edge_index(dataset_name, data_dir, device):
    """加载GNN边索引（优先 data/processed，再尝试 data/embeddings/processed）

    Returns:
        edge_index (torch.Tensor or None): 边索引张量
        max_kc_id (int or None): edge_index 中最大的 kc_id
    """
    candidate_paths = [
        os.path.join(data_dir, "processed", f"{dataset_name}_edge_index.npy"),
        os.path.join(data_dir, "embeddings", "processed", f"{dataset_name}_edge_index.npy"),
    ]

    edge_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if edge_path is None:
        return None, None

    edge_index_np = np.load(edge_path)
    if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        print(f"⚠️  edge_index格式异常: {edge_path}, shape={edge_index_np.shape}")
        return None, None

    edge_index = torch.from_numpy(edge_index_np).long().to(device)
    max_kc_id = int(edge_index_np.max())
    print(f"✓ 加载edge_index: {edge_path}, 边数={edge_index.size(1)}, max_kc_id={max_kc_id}")
    return edge_index, max_kc_id


def load_precomputed_embeddings(data_dir, use_gnn=False):
    """加载预计算嵌入对象"""
    emb_dir = os.path.join(data_dir, "embeddings")
    precomputed = PrecomputedEmbeddings(embedding_dir=emb_dir)

    precomputed.load_question_embeddings()
    if use_gnn:
        try:
            precomputed.load_kc_embeddings()
        except FileNotFoundError:
            print("⚠️  未找到知识点预计算嵌入，将仅使用题目预计算嵌入")

    return precomputed


def train_epoch(model, train_data, optimizer, config, q_texts=None, tokenizer=None,
                q_to_kc_mapping=None, edge_index=None, scaler=None, use_amp=False):
    """训练一个epoch"""
    model.train()
    device = torch.device(config.get('device', config.get('training_device', 'cuda')))

    total_loss = 0.0
    total_cnt = 0
    seq_len = config.get('seq_len', None)  # 从 config 读取 seq_len

    # 梯度累积
    gradient_accumulation_steps = config.get('gradient_accumulation_steps',
                                              config.get('training_gradient_accumulation_steps', 1))
    use_cl_loss = config.get('cl_loss', config.get('recommendation_cl_loss', False))

    # KTData.__len__ 返回样本数，进度条应显示 batch 数（DataLoader 长度）
    total_batches = len(train_data.loader)
    pbar = tqdm(train_data, total=total_batches, desc="训练")
    processed_batches = 0

    for batch_idx, batch in enumerate(pbar):
        processed_batches += 1
        # 准备数据
        if config.get('training_with_pid', config.get('with_pid', False)):
            q, s, pid = batch.get("q", "s", "pid")
        else:
            q, s = batch.get("q", "s")
            pid = None

        # 如果 seq_len 为 None，需要包装成 list
        if seq_len is None:
            q, s, pid = [q], [s], [pid]

        # 准备LLM文本输入
        q_text_input = None
        if config.get('use_llm', False) and q_texts is not None:
            q_text_input = prepare_bert_inputs(q_texts, q, tokenizer)

        # 准备GNN输入
        kc_ids = None
        if config.get('use_gnn', False) and q_to_kc_mapping is not None:
            n_kc = config.get('gnn_n_kc', config.get('n_kc', None))
            kc_ids = add_kc_ids_to_batch(q, q_to_kc_mapping, n_kc)
            # 注意：add_kc_ids_to_batch 已经处理了 list 情况

        # 如果是seq_len模式，复制输入
        if q_text_input is not None and seq_len is None:
            q_text_input = [q_text_input]

        for idx, (q, s, pid) in enumerate(zip(q, s, pid)):
            q = q.to(device)
            s = s.to(device)
            if pid is not None:
                pid = pid.to(device)

            # 处理文本输入
            current_q_text = None
            if q_text_input is not None:
                current_q_text = q_text_input[idx] if len(q_text_input) > 1 else q_text_input[0]
                current_q_text = {k: v.to(device) for k, v in current_q_text.items()}

            # 处理kc_ids
            current_kc_ids = None
            if kc_ids is not None:
                # 简化逻辑
                if isinstance(kc_ids, list):
                    current_kc_ids = kc_ids[idx] if len(kc_ids) > idx else kc_ids[0]
                else:
                    current_kc_ids = kc_ids

                # 确保 current_kc_ids 是 tensor 并移到设备
                if isinstance(current_kc_ids, torch.Tensor):
                    current_kc_ids = current_kc_ids.to(device)
                else:
                    # 如果还是 list，递归处理
                    current_kc_ids = current_kc_ids[idx] if len(current_kc_ids) > idx else current_kc_ids[0]
                    if isinstance(current_kc_ids, torch.Tensor):
                        current_kc_ids = current_kc_ids.to(device)

            # 计算损失
            # 处理 DataParallel 包装的模型
            model_to_use = model.module if hasattr(model, 'module') else model
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                if use_cl_loss:
                    loss_out = model_to_use.get_cl_loss(
                        q, s, pid, current_kc_ids, edge_index, current_q_text, None
                    )
                    loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
                else:
                    loss = model_to_use.get_loss(
                        q, s, pid, current_kc_ids, edge_index, current_q_text, None
                    )

            # 梯度累积：损失除以累积步数
            loss = loss / gradient_accumulation_steps

            # 反向传播
            # 处理 DataParallel 的参数
            params_to_clip = model.module.parameters() if hasattr(model, 'module') else model.parameters()

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 每累积一定步数后更新参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps  # 恢复原始损失值
            total_cnt += 1

            pbar.set_postfix({'loss': total_loss / total_cnt})

    # 处理最后可能未更新的梯度
    if (processed_batches % gradient_accumulation_steps) != 0:
        params_to_clip = model.module.parameters() if hasattr(model, 'module') else model.parameters()
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_cnt


def validate(model, valid_data, config, q_texts=None, tokenizer=None,
             q_to_kc_mapping=None, edge_index=None, use_amp=False):
    """验证模型"""
    model.eval()
    device = torch.device(config.get('device', config.get('training_device', 'cuda')))
    evaluator = Evaluator()
    seq_len = config.get('seq_len', None)  # 从 config 读取 seq_len

    with torch.no_grad():
        total_batches = len(valid_data.loader)
        for batch in tqdm(valid_data, total=total_batches, desc="验证"):
            if config.get('training_with_pid', config.get('with_pid', False)):
                q, s, pid = batch.get("q", "s", "pid")
            else:
                q, s = batch.get("q", "s")
                pid = None

            # 如果 seq_len 为 None，需要包装成 list
            if seq_len is None:
                q, s, pid = [q], [s], [pid]

            # 准备输入（同训练）
            q_text_input = None
            if config.get('use_llm', False) and q_texts is not None:
                q_text_input = prepare_bert_inputs(q_texts, q, tokenizer)
                if seq_len is None:
                    q_text_input = [q_text_input]

            kc_ids = None
            if config.get('use_gnn', False) and q_to_kc_mapping is not None:
                kc_ids = add_kc_ids_to_batch(q, q_to_kc_mapping)
                # 注意：add_kc_ids_to_batch 已经处理了 list 情况

            for idx, (q, s, pid) in enumerate(zip(q, s, pid)):
                q = q.to(device)
                s = s.to(device)
                if pid is not None:
                    pid = pid.to(device)

                current_q_text = None
                if q_text_input is not None:
                    current_q_text = q_text_input[idx] if len(q_text_input) > 1 else q_text_input[0]
                    current_q_text = {k: v.to(device) for k, v in current_q_text.items()}

                current_kc_ids = None
                if kc_ids is not None:
                    # 简化逻辑
                    if isinstance(kc_ids, list):
                        current_kc_ids = kc_ids[idx] if len(kc_ids) > idx else kc_ids[0]
                    else:
                        current_kc_ids = kc_ids

                    # 确保 current_kc_ids 是 tensor 并移到设备
                    if isinstance(current_kc_ids, torch.Tensor):
                        current_kc_ids = current_kc_ids.to(device)
                    else:
                        # 如果还是 list，递归处理
                        current_kc_ids = current_kc_ids[idx] if len(current_kc_ids) > idx else current_kc_ids[0]
                        if isinstance(current_kc_ids, torch.Tensor):
                            current_kc_ids = current_kc_ids.to(device)

                # 处理 DataParallel 包装的模型
                model_to_use = model.module if hasattr(model, 'module') else model
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    y, *_ = model_to_use.predict(q, s, pid, current_kc_ids, edge_index, current_q_text, None)
                evaluator.evaluate(s, torch.sigmoid(y))

    return evaluator.report()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SGSAT-KT 训练脚本")
    parser.add_argument('mode', choices=['test', 'baseline', 'full', 'prod'], help='训练模式')
    parser.add_argument('--device', help='覆盖设备设置')
    args = parser.parse_args()

    # 统一从 default.yaml 加载配置
    try:
        loader = ConfigLoader()
        default_config = loader.load_yaml('default.yaml')

        # 应用预设配置
        presets = default_config.get('presets', {})
        if args.mode in presets:
            # 合并预设配置到基础配置
            import copy
            base_config = copy.deepcopy(default_config)
            # 移除 presets 避免递归
            if 'presets' in base_config:
                del base_config['presets']
            # 递归合并
            config = merge_dicts(base_config, presets[args.mode])
        else:
            config = default_config

        print(f"✅ 从 configs/default.yaml 加载配置")
        print(f"📝 使用预设: {args.mode}")

    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.device:
        config['training']['device'] = args.device

    # 扁平化配置，方便后续访问
    config = flatten_config(config)

    # 打印配置
    print("=" * 60)
    print(f"🚀 训练模式: {args.mode}")
    print("=" * 60)
    # 打印主要配置
    print(f"  批大小: {config.get('training_batch_size', config.get('batch_size', 'N/A'))}")
    print(f"  学习率: {config.get('training_learning_rate', config.get('learning_rate', 'N/A'))}")
    print(f"  训练轮数: {config.get('training_n_epochs', config.get('epochs', 'N/A'))}")
    print(f"  使用LLM: {config.get('llm_use_llm', config.get('use_llm', 'N/A'))}")
    print(f"  使用GNN: {config.get('gnn_use_gnn', config.get('use_gnn', 'N/A'))}")
    print(f"  GPU设备: {config.get('gpu_device_ids', 'N/A')}")
    print("=" * 60)

    # 检查预计算嵌入
    if config.get('use_precomputed', False):
        emb_path = os.path.join(project_root, 'data/embeddings/question_embeddings.pkl')
        if not os.path.exists(emb_path):
            print("\n❌ 预计算嵌入不存在！")
            print("请先运行: ./scripts/1_precompute.sh")
            sys.exit(1)
        print("✅ 预计算嵌入检查通过")

    # 加载数据
    print("\n📂 加载数据...")
    import tomlkit
    datasets = tomlkit.load(open(os.path.join(project_root, "data/datasets.toml")))
    dataset_name = config.get('training_dataset', config.get('dataset', 'xes'))
    dataset_config = datasets[dataset_name]

    data_dir = os.path.join(project_root, "data")
    seq_len = dataset_config.get("seq_len", None)
    config['seq_len'] = seq_len  # 添加到 config 中

    batch_size = config.get('batch_size', config.get('training_batch_size', 16))
    train_data = KTData(
        os.path.join(data_dir, dataset_config["train"]),
        dataset_config["inputs"],
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
    )

    test_batch_size = config.get('test_batch_size', config.get('training_test_batch_size', 8))
    valid_data = KTData(
        os.path.join(data_dir, dataset_config.get("valid", dataset_config["test"])),
        dataset_config["inputs"],
        seq_len=seq_len,
        batch_size=test_batch_size,
    )
    print(f"✅ 数据加载完成: {dataset_name}")

    # 加载文本数据（如果使用LLM）
    q_texts = None
    tokenizer = None
    if config.get('use_llm', False) and not config.get('use_precomputed', False):
        q_texts = load_text_data(dataset_name, data_dir)
        if q_texts is None:
            print("⚠️  无法加载文本数据，关闭LLM")
            config['use_llm'] = False
        else:
            from transformers import AutoTokenizer
            model_path = config.get('pretrained_model', 'bert-base-chinese')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"✓ 加载tokenizer: {model_path}")

    # 加载q->kc映射（如果使用GNN）
    q_to_kc_mapping = None
    edge_index = None
    if config.get('use_gnn', False):
        q_to_kc_mapping = load_q_to_kc_mapping(dataset_name, data_dir)
        if q_to_kc_mapping is None:
            print("⚠️  无法加载q->kc映射，关闭GNN")
            config['use_gnn'] = False
        else:
            # 自动校正 n_kc，避免配置值过小导致索引越界
            inferred_n_kc = max(q_to_kc_mapping.values()) + 1 if q_to_kc_mapping else 0
            configured_n_kc = config.get('gnn_n_kc', config.get('n_kc', 100))
            if inferred_n_kc > configured_n_kc:
                print(f"⚠️  检测到 n_kc 配置过小({configured_n_kc})，自动提升到 {inferred_n_kc}")
                config['n_kc'] = inferred_n_kc
                config['gnn_n_kc'] = inferred_n_kc

            edge_index, edge_max_kc = load_edge_index(dataset_name, data_dir, torch.device('cpu'))
            if edge_index is None:
                print("⚠️  未找到edge_index，GNN先决图分支将被跳过")
            elif edge_max_kc is not None:
                # 检查 edge_index 中的最大 kc_id 是否超出 n_kc
                current_n_kc = config.get('gnn_n_kc', config.get('n_kc', 100))
                if edge_max_kc >= current_n_kc:
                    required_n_kc = edge_max_kc + 1
                    print(f"⚠️  edge_index 包含 kc_id={edge_max_kc}，超出 n_kc={current_n_kc}")
                    print(f"🔧 自动提升 n_kc: {current_n_kc} -> {required_n_kc}")
                    config['n_kc'] = required_n_kc
                    config['gnn_n_kc'] = required_n_kc

    # 加载预计算嵌入（仅在 LLM 分支启用时有意义）
    precomputed_embeddings = None
    if config.get('use_precomputed', False) and config.get('use_llm', False):
        try:
            precomputed_embeddings = load_precomputed_embeddings(
                data_dir,
                use_gnn=config.get('use_gnn', False),
            )
            print("✅ 预计算嵌入已加载到训练流程")
        except Exception as e:
            print(f"⚠️  预计算嵌入加载失败，回退到在线文本分支: {e}")
            precomputed_embeddings = None
            config['use_precomputed'] = False
    elif config.get('use_precomputed', False) and not config.get('use_llm', False):
        print("ℹ️  use_llm=False，跳过预计算嵌入加载")
        config['use_precomputed'] = False

    # 预计算回退：若关闭了预计算且尚未准备在线文本输入，则补充加载
    if config.get('use_llm', False) and not config.get('use_precomputed', False) and q_texts is None:
        q_texts = load_text_data(dataset_name, data_dir)
        if q_texts is None:
            print("⚠️  回退在线文本分支失败：文本数据缺失，LLM分支将退化为ID嵌入")
        else:
            from transformers import AutoTokenizer
            model_path = config.get('pretrained_model', 'bert-base-chinese')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"✓ 回退成功，已加载在线tokenizer: {model_path}")

    if config.get('use_graph_similarity', config.get('recommendation_use_graph_similarity', False)):
        print("⚠️  use_graph_similarity 已开启，但当前训练主流程尚未接入 DCFSimGraphEnhanced")

    # 创建模型
    print("\n🤖 创建模型...")
    # 确保 n_kc 配置正确
    final_n_kc = config.get('gnn_n_kc', config.get('n_kc', 100))
    print(f"📊 使用 n_kc={final_n_kc} 创建模型")
    if edge_index is not None:
        print(f"📊 edge_index max_kc={edge_index.max().item()}")
    model = DTransformer(
        dataset_config["n_questions"],
        dataset_config.get("n_pid", 0),
        d_model=config.get('model_d_model', config.get('d_model', 256)),
        d_fc=config.get('model_d_fc', config.get('d_fc', 512)),
        n_heads=config.get('model_n_heads', config.get('n_heads', 8)),
        n_layers=config.get('model_n_layers', config.get('n_layers', 2)),
        n_know=config.get('model_n_know', config.get('n_know', 32)),
        lambda_cl=config.get('recommendation_lambda_cl', config.get('lambda_cl', 0.1)),
        dropout=config.get('model_dropout', config.get('dropout', 0.2)),
        proj=config.get('proj', False),
        hard_neg=config.get('hard_neg', False),
        window=config.get('model_window', config.get('window', 1)),
        use_llm=config.get('use_llm', False),
        pretrained_model=config.get('pretrained_model', 'bert-base-chinese'),
        precomputed_embeddings=precomputed_embeddings,
        id_dim=config.get('model_id_dim', config.get('id_dim', 128)),
        llm_proj_dim=config.get('model_llm_proj_dim', config.get('llm_proj_dim', 256)),
        llm_inter_dim=config.get('model_llm_inter_dim', config.get('llm_inter_dim', 512)),
        use_gnn=config.get('use_gnn', False),
        n_kc=final_n_kc,
        gnn_layers=config.get('gnn_gnn_layers', config.get('gnn_layers', 2)),
    )
    print(f"✅ 模型创建完成，实际 n_kc={model.n_kc}")

    # 分支激活报告：避免配置开启但链路未生效
    llm_enabled = bool(config.get('use_llm', False))
    precomputed_active = bool(config.get('use_precomputed', False) and precomputed_embeddings is not None)
    online_text_active = bool(llm_enabled and q_texts is not None and tokenizer is not None)
    gnn_enabled = bool(config.get('use_gnn', False))
    gnn_active = bool(gnn_enabled and edge_index is not None)
    cl_loss_active = bool(config.get('cl_loss', config.get('recommendation_cl_loss', False)))

    print("📌 分支激活状态:")
    print(f"  use_llm: {llm_enabled}")
    print(f"  use_precomputed: {precomputed_active}")
    print(f"  use_online_text: {online_text_active}")
    print(f"  use_gnn: {gnn_active}")
    print(f"  use_cl_loss: {cl_loss_active}")

    if gnn_enabled and edge_index is not None:
        print(f"  gnn_edges: {edge_index.size(1)}")

    print("📌 分支诊断:")
    if not llm_enabled:
        print("  LLM分支未启用：当前配置 use_llm=False（test/baseline 预设默认关闭）")
    else:
        if not precomputed_active and not online_text_active:
            print("  LLM分支退化为ID：预计算未加载且在线文本未就绪")
        elif precomputed_active:
            print("  LLM分支生效：使用预计算嵌入")
        elif online_text_active:
            print("  LLM分支生效：使用在线文本编码")

    if not gnn_enabled:
        print("  GNN分支未启用：当前配置 use_gnn=False（test/baseline 预设默认关闭）")
    elif edge_index is None:
        print("  GNN分支未生效：未找到 edge_index 文件")
    else:
        print("  GNN分支生效：edge_index 已加载")

    if not cl_loss_active:
        print("  CL损失未启用：recommendation.cl_loss=False")
    else:
        print("  CL损失已启用：训练将调用 get_cl_loss")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('training_l2', config.get('l2', 1e-5))
    )

    # 设置设备和多GPU
    device = torch.device(config.get('device', config.get('training_device', 'cuda')))

    # 检查是否使用多GPU
    gpu_device_ids = config.get('gpu_device_ids', None)
    use_multi_gpu = (
        gpu_device_ids is not None
        and len(gpu_device_ids) > 1
        and str(device).startswith('cuda')
    )

    if use_multi_gpu:
        print(f"🚀 使用多GPU训练: {gpu_device_ids}")
        # 检查所有GPU是否可用
        available_gpus = []
        for gpu_id in gpu_device_ids:
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                available_gpus.append(gpu_id)
            else:
                print(f"⚠️  GPU {gpu_id} 不可用，跳过")

        if len(available_gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=available_gpus)
            device = torch.device(f'cuda:{available_gpus[0]}')
            batch_size = config.get('batch_size', config.get('training_batch_size', 16))
            grad_accum = config.get('gradient_accumulation_steps',
                                   config.get('training_gradient_accumulation_steps', 1))
            print(f"✅ DataParallel已启用")
            print(f"   主设备: cuda:{available_gpus[0]}")
            print(f"   所有设备: {available_gpus}")
            print(f"   批大小: {batch_size}")
            print(f"   梯度累积步数: {grad_accum}")
            print(f"   有效批大小: {batch_size * grad_accum}")
            print(f"   每GPU批大小约: {batch_size // len(available_gpus)}")
            print(f"💡 提示: DataParallel会在所有GPU上计算，但主卡负载会更高")
        elif len(available_gpus) == 1:
            print(f"⚠️  只有1个可用GPU，使用单GPU模式")
            device = torch.device(f'cuda:{available_gpus[0]}')
        else:
            print("⚠️  没有可用GPU，使用CPU")
            device = torch.device('cpu')
    else:
        print(f"📱 使用设备: {device}")

    model.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)

    amp_flag = config.get('use_amp', config.get('training_use_amp', True))
    use_amp = bool(amp_flag) and str(device).startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"⚙️  AMP混合精度: {'开启' if use_amp else '关闭'}")

    # 创建输出目录
    output_dir = None
    if config.get('training_save_model', config.get('save_model', True)):
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, f"output/{args.mode}_{dataset_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # 保存配置
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📁 输出目录: {output_dir}")

    # 训练循环
    epochs = config.get('epochs', config.get('n_epochs', 30))
    print(f"\n🏋️  开始训练 {epochs} epochs...")
    best = {"auc": 0}
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")

        # 训练
        avg_loss = train_epoch(
            model,
            train_data,
            optimizer,
            config,
            q_texts,
            tokenizer,
            q_to_kc_mapping,
            edge_index,
            scaler,
            use_amp,
        )
        print(f"训练损失: {avg_loss:.4f}")

        # 验证
        print("验证中...")
        results = validate(
            model,
            valid_data,
            config,
            q_texts,
            tokenizer,
            q_to_kc_mapping,
            edge_index,
            use_amp,
        )
        print(f"验证结果: {results}")

        # 保存最佳模型
        if results["auc"] > best["auc"]:
            best = results
            best_epoch = epoch

            if output_dir:
                model_path = os.path.join(output_dir, f"best_model.pt")
                # 处理 DataParallel 包装的模型
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, model_path)
                print(f"💾 保存最佳模型: {model_path}")

        # 早停
        early_stop = config.get('training_early_stop', config.get('early_stop', 0))
        if early_stop > 0 and epoch - best_epoch >= early_stop:
            print(f"⏸️  早停触发 (Best: Epoch {best_epoch})")
            break

    print(f"\n{'='*60}")
    print(f"🎉 训练完成!")
    print(f"{'='*60}")
    print(f"最佳结果 (Epoch {best_epoch}):")
    for k, v in best.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
