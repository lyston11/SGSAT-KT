"""
数据预处理脚本 - 为SGSAT-KT准备GNN先决图数据
从训练数据中提取知识点共现关系，构建先决图
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict, Counter

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_kt_data(file_path):
    """加载KT数据"""
    sequences = []
    with open(file_path, 'r') as f:
        while True:
            # 读取序列长度
            line = f.readline()
            if not line:
                break
            seq_len = int(line.strip())

            # 读取pid序列（跳过）
            f.readline()

            # 读取知识点序列
            kc_line = f.readline().strip()
            kc_sequence = [int(x) for x in kc_line.split(",")]

            # 读取作答序列
            s_line = f.readline().strip()
            s_sequence = [int(x) for x in s_line.split(",")]

            sequences.append({
                'kc_sequence': kc_sequence,
                's_sequence': s_sequence
            })

    return sequences


def extract_kc_info(sequences, min_cooccurrence=5):
    """
    提取知识点信息并构建先决图

    策略：
    1. 统计知识点共现频率（在一个时间窗口内同时出现）
    2. 如果知识点A在知识点B之前频繁出现，认为A是B的先决条件
    """
    print("提取知识点信息...")

    # 统计知识点
    kc_counter = Counter()
    for seq in sequences:
        kc_counter.update(seq['kc_sequence'])

    n_kc = len(kc_counter)
    print(f"发现 {n_kc} 个不同的知识点")

    # 提取知识点ID映射
    kc_ids = sorted(kc_counter.keys())
    kc_to_idx = {kc_id: idx for idx, kc_id in enumerate(kc_ids)}

    # 构建先决图边
    # 策略：如果知识点A经常在知识点B之前出现，则添加边 A->B
    edge_weights = defaultdict(int)

    window_size = 5  # 时间窗口大小

    for seq in sequences:
        kc_seq = seq['kc_sequence']
        for i in range(len(kc_seq)):
            for j in range(i + 1, min(i + window_size + 1, len(kc_seq))):
                kc_a = kc_seq[i]
                kc_b = kc_seq[j]

                if kc_a in kc_to_idx and kc_b in kc_to_idx:
                    idx_a = kc_to_idx[kc_a]
                    idx_b = kc_to_idx[kc_b]
                    edge_weights[(idx_a, idx_b)] += 1

    # 过滤低频边
    min_weight = min_cooccurrence
    edges = [(src, dst) for (src, dst), weight in edge_weights.items() if weight >= min_weight]

    print(f"构建了 {len(edges)} 条先决图边（共现次数 >= {min_weight}）")

    # 转换为torch格式的edge_index
    if len(edges) > 0:
        edge_index = np.array(edges).T  # shape: [2, n_edges]
    else:
        # 如果没有边，创建一个空图
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return kc_ids, edge_index, n_kc


def save_preprocessed_data(kc_ids, edge_index, n_kc, output_dir, dataset_name):
    """保存预处理数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存知识点列表
    kc_file = os.path.join(output_dir, f"{dataset_name}_kc_ids.npy")
    np.save(kc_file, np.array(kc_ids, dtype=np.int64))
    print(f"保存知识点ID到: {kc_file}")

    # 保存边索引
    edge_file = os.path.join(output_dir, f"{dataset_name}_edge_index.npy")
    np.save(edge_file, edge_index)
    print(f"保存边索引到: {edge_file}")

    # 保存元信息
    meta_file = os.path.join(output_dir, f"{dataset_name}_meta.json")
    meta = {
        'n_kc': n_kc,
        'n_edges': edge_index.shape[1] if len(edge_index.shape) > 1 else 0,
        'kc_ids_file': kc_file,
        'edge_index_file': edge_file
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"保存元信息到: {meta_file}")

    return kc_file, edge_file


def main():
    parser = argparse.ArgumentParser(description='预处理KT数据以支持GNN')
    parser.add_argument('--dataset', type=str, default='assist09',
                        choices=['assist09', 'assist17', 'statics', 'algebra05', 'doudouyun'],
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录')
    parser.add_argument('--min_cooccurrence', type=int, default=5,
                        help='最小共现次数阈值')

    args = parser.parse_args()

    # 构建路径
    data_dir = os.path.join(project_root, args.data_dir)
    train_file = os.path.join(data_dir, f"{args.dataset}/train.txt")

    print(f"加载数据集: {args.dataset}")
    print(f"训练文件: {train_file}")

    # 加载数据
    sequences = load_kt_data(train_file)
    print(f"加载了 {len(sequences)} 个序列")

    # 提取知识点信息并构建先决图
    kc_ids, edge_index, n_kc = extract_kc_info(sequences, args.min_cooccurrence)

    # 保存预处理数据
    save_preprocessed_data(kc_ids, edge_index, n_kc,
                          os.path.join(project_root, args.output_dir),
                          args.dataset)

    print("\n✅ 数据预处理完成！")
    print(f"知识点数量: {n_kc}")
    print(f"边数量: {edge_index.shape[1] if len(edge_index.shape) > 1 else 0}")
    print(f"\n使用方法:")
    print(f"在训练脚本中加载：")
    print(f"  kc_ids = np.load('data/processed/{args.dataset}_kc_ids.npy')")
    print(f"  edge_index = np.load('data/processed/{args.dataset}_edge_index.npy')")


if __name__ == '__main__':
    main()
