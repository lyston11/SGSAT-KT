"""
XES数据集预处理脚本
将XES数据转换为SGSAT-KT格式，并生成GNN边索引和文本数据
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)


def load_xes_data(file_path, max_rows=None):
    """加载XES数据集"""
    print(f"📚 加载XES数据集: {file_path}")
    df = pd.read_csv(file_path, nrows=max_rows)

    print(f"✅ 加载完成")
    print(f"   总记录数: {len(df):,}")
    print(f"   题目数: {df['item_id'].nunique():,}")
    print(f"   学生数: {df['user_id'].nunique():,}")
    print(f"   知识点数: {df['skill_name'].nunique():,}")

    return df


def filter_xes_data(df):
    """过滤无效记录，保证后续映射和训练数据一致。"""
    required_columns = ["user_id", "item_id", "correct", "skill_name", "order_id"]
    df = df.dropna(subset=required_columns).copy()
    df = df[df["correct"].isin([0, 1])].copy()
    df = df[df["skill_name"] != -1].copy()

    # 规范类型，避免后续排序和映射不稳定
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["correct"] = df["correct"].astype(int)
    df["skill_name"] = df["skill_name"].astype(int)
    df["order_id"] = df["order_id"].astype(np.int64)

    print(f"\n🧹 过滤后数据:")
    print(f"   总记录数: {len(df):,}")
    print(f"   题目数: {df['item_id'].nunique():,}")
    print(f"   学生数: {df['user_id'].nunique():,}")
    print(f"   知识点数: {df['skill_name'].nunique():,}")

    return df


def create_kc_mappings(df):
    """创建知识点映射"""
    # 获取所有唯一的知识点
    unique_skills = df['skill_name'].unique()
    n_skills = len(unique_skills)

    # 创建知识点ID映射 (skill_name -> kc_id)
    skill_to_kc = {skill: idx for idx, skill in enumerate(sorted(unique_skills))}

    # 获取所有唯一的题目
    unique_items = df['item_id'].unique()
    n_items = len(unique_items)

    # 创建题目ID映射 (item_id -> q_id)
    item_to_q = {item: idx for idx, item in enumerate(sorted(unique_items))}

    # 创建题目-知识点映射
    item_skills = df.groupby('item_id')['skill_name'].apply(lambda x: list(set(x))).to_dict()

    print(f"\n📝 映射信息:")
    print(f"   知识点数量: {n_skills}")
    print(f"   题目数量: {n_items}")

    return skill_to_kc, item_to_q, item_skills, n_skills, n_items


def convert_to_sgsakt_format(df, output_file, item_to_q, max_seq_len=200):
    """将XES数据转换为标准 KT 格式: seq_len, q_ids, scores"""
    print(f"\n🔄 转换数据格式...")

    # 按学生分组
    student_sequences = df.groupby('user_id')

    # 准备输出数据
    output_lines = []

    processed = 0
    for user_id, group in student_sequences:
        # 按order_id排序
        group = group.sort_values('order_id')

        # 提取序列
        q_ids = [item_to_q[item_id] for item_id in group['item_id']]
        scores = group['correct'].values

        # 如果序列太长，分割成多个子序列
        for i in range(0, len(q_ids), max_seq_len):
            q_seq = q_ids[i:i+max_seq_len]
            s_seq = scores[i:i+max_seq_len]

            if len(q_seq) >= 5:  # 至少5个题目
                # 格式: seq_len, q_ids, scores
                output_lines.append(str(len(q_seq)))
                output_lines.append(','.join(map(str, q_seq)))
                output_lines.append(','.join(map(str, s_seq)))

        processed += 1
        if processed % 1000 == 0:
            print(f"   处理进度: {processed}/{len(student_sequences)} 学生")

    # 保存训练数据
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"✅ 转换完成，保存到: {output_file}")
    print(f"   序列数量: {len(output_lines) // 3}")


def split_train_valid_test(df, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """按学生划分 train/valid/test。"""
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio 和 valid_ratio 配置非法")

    user_ids = df['user_id'].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(user_ids)

    n_train = int(len(user_ids) * train_ratio)
    n_valid = int(len(user_ids) * valid_ratio)

    train_users = set(user_ids[:n_train])
    valid_users = set(user_ids[n_train:n_train + n_valid])
    test_users = set(user_ids[n_train + n_valid:])

    train_df = df[df['user_id'].isin(train_users)]
    valid_df = df[df['user_id'].isin(valid_users)]
    test_df = df[df['user_id'].isin(test_users)]

    print(f"\n📊 数据分割:")
    print(f"   训练集: {len(train_df):,} 记录 ({len(train_users)} 学生)")
    print(f"   验证集: {len(valid_df):,} 记录 ({len(valid_users)} 学生)")
    print(f"   测试集: {len(test_df):,} 记录 ({len(test_users)} 学生)")

    return train_df, valid_df, test_df


def build_prerequisite_graph(df, skill_to_kc, item_to_q, item_skills, min_cooccurrence=10):
    """构建知识点先决图"""
    print(f"\n🕸️ 构建知识点先决图...")

    # 收集题目中的知识点序列
    kc_sequences = []
    for user_id, group in df.groupby('user_id'):
        group = group.sort_values('order_id')
        kc_seq = [skill_to_kc[skill] for skill in group['skill_name']]
        kc_sequences.append(kc_seq)

    # 统计知识点共现
    window_size = 5
    edge_weights = defaultdict(int)

    for kc_seq in kc_sequences:
        for i in range(len(kc_seq)):
            for j in range(i + 1, min(i + window_size + 1, len(kc_seq))):
                kc_a = kc_seq[i]
                kc_b = kc_seq[j]
                edge_weights[(kc_a, kc_b)] += 1

    # 过滤低频边
    edges = [(src, dst) for (src, dst), weight in edge_weights.items() if weight >= min_cooccurrence]

    # 转换为numpy数组
    if len(edges) > 0:
        edge_index = np.array(edges).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    print(f"✅ 先决图构建完成:")
    print(f"   知识点数量: {len(skill_to_kc)}")
    print(f"   边数量: {len(edges)}")

    return edge_index


def save_text_data(df, item_to_q, skill_to_kc, item_skills, output_dir):
    """保存文本数据"""
    print(f"\n📝 生成文本数据...")

    os.makedirs(output_dir, exist_ok=True)

    # 为每个题目生成文本
    question_texts = {}
    for item_id, q_id in item_to_q.items():
        skills = item_skills.get(item_id, [])
        if skills:
            primary_skill = int(skills[0])
            kc_id = int(skill_to_kc[primary_skill])
            # 使用实际的题目内容
            item_rows = df[df['item_id'] == item_id]
            if len(item_rows) > 0:
                content = str(item_rows['content'].values[0])
                # 清理内容（移除图片引用等）
                content_clean = content.split('question_')[0].strip()
                if not content_clean:
                    content_clean = f"题目{item_id}"
            else:
                content_clean = f"题目{item_id}（知识点：{primary_skill}）"

            question_texts[str(int(q_id))] = {
                "content": content_clean,
                "skill": str(kc_id),
                "raw_skill": str(primary_skill),
                "item_id": str(int(item_id)),
            }

    # 为每个知识点生成文本
    kc_texts = {}
    for raw_skill_name, kc_id in skill_to_kc.items():
        kc_id_int = int(kc_id)
        kc_texts[str(kc_id_int)] = {
            "name": str(raw_skill_name),
            "description": f"知识点：{raw_skill_name}",
            "raw_skill": str(int(raw_skill_name)),
        }

    # 保存文本数据
    q_text_file = os.path.join(output_dir, "xes_question_texts.json")
    kc_text_file = os.path.join(output_dir, "xes_kc_texts.json")

    with open(q_text_file, 'w', encoding='utf-8') as f:
        json.dump(question_texts, f, ensure_ascii=False, indent=2)

    with open(kc_text_file, 'w', encoding='utf-8') as f:
        json.dump(kc_texts, f, ensure_ascii=False, indent=2)

    print(f"✅ 文本数据已保存:")
    print(f"   题目文本: {q_text_file} ({len(question_texts)} 条)")
    print(f"   知识点文本: {kc_text_file} ({len(kc_texts)} 条)")


def save_metadata(n_questions, n_kc, n_students, output_dir, dataset_name="xes"):
    """保存元数据"""
    datasets_config = {
        "xes": {
            "train": "xes/train.txt",
            "valid": "xes/valid.txt",
            "test": "xes/test.txt",
            "n_questions": n_questions,
            "n_pid": 0,
            "inputs": ["q", "s"],
            "seq_len": 200,
            "has_text": True,
        }
    }

    # 保存到datasets.toml
    output_file = os.path.join(output_dir, "datasets_xes.toml")
    with open(output_file, 'w') as f:
        import tomlkit
        tomlkit.dump(datasets_config, f)

    print(f"✅ 元数据已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='处理XES数据集用于SGSAT-KT训练')
    parser.add_argument('--input', type=str, default='data/xes/xes_math.csv',
                        help='XES数据文件路径')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='输出目录')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='最大处理行数（用于测试）')
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='最大序列长度')
    parser.add_argument('--min_cooccurrence', type=int, default=10,
                        help='知识点共现最小次数（用于构建先决图）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集用户比例')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='验证集用户比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    input_path = os.path.join(project_root, args.input)
    output_dir = os.path.join(project_root, args.output_dir)

    print("=" * 60)
    print("  XES数据集预处理 - SGSAT-KT")
    print("=" * 60)

    # 1. 加载数据
    df = load_xes_data(input_path, args.max_rows)
    df = filter_xes_data(df)

    # 2. 创建映射
    skill_to_kc, item_to_q, item_skills, n_skills, n_items = create_kc_mappings(df)

    # 3. 分割训练集和测试集
    train_df, valid_df, test_df = split_train_valid_test(
        df,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    # 4. 转换为SGSAT-KT格式
    train_output = os.path.join(output_dir, "xes", "train.txt")
    valid_output = os.path.join(output_dir, "xes", "valid.txt")
    test_output = os.path.join(output_dir, "xes", "test.txt")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(train_output), exist_ok=True)

    convert_to_sgsakt_format(train_df, train_output, item_to_q, args.max_seq_len)
    convert_to_sgsakt_format(valid_df, valid_output, item_to_q, args.max_seq_len)
    convert_to_sgsakt_format(test_df, test_output, item_to_q, args.max_seq_len)

    # 5. 构建先决图
    edge_index = build_prerequisite_graph(train_df, skill_to_kc, item_to_q, item_skills, args.min_cooccurrence)

    # 保存边索引
    edge_file = os.path.join(output_dir, "processed", "xes_edge_index.npy")
    kc_list_file = os.path.join(output_dir, "processed", "xes_kc_ids.npy")

    os.makedirs(os.path.dirname(edge_file), exist_ok=True)

    kc_ids_list = list(range(n_skills))
    np.save(edge_file, edge_index)
    np.save(kc_list_file, np.array(kc_ids_list, dtype=np.int64))

    print(f"✅ GNN数据已保存:")
    print(f"   边索引: {edge_file}")
    print(f"   知识点ID: {kc_list_file}")

    # 6. 保存文本数据
    text_output_dir = os.path.join(output_dir, "text_data")
    save_text_data(df, item_to_q, skill_to_kc, item_skills, text_output_dir)

    # 7. 保存元数据
    save_metadata(n_items, n_skills, df['user_id'].nunique(), output_dir)

    print("\n" + "=" * 60)
    print("  ✅ 数据预处理完成！")
    print("=" * 60)
    print(f"\n📊 数据统计:")
    print(f"   题目数量: {n_items:,}")
    print(f"   知识点数量: {n_skills:,}")
    print(f"   学生数量: {df['user_id'].nunique():,}")
    print(f"\n💡 使用方法:")
    print(f"   ./scripts/1_precompute.sh")
    print(f"   ./scripts/train.sh full")


if __name__ == '__main__':
    main()
