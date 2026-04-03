"""
数据预处理脚本 - 为 TriSG-KT 准备多数据集图结构和最小文本数据

目标：
1. 按 data/datasets.toml 中的 inputs 配置读取不同布局的 KT 序列文件
2. 基于 q 序列构建可供 GNN 使用的知识点先决图
3. 为缺少文本描述的数据集生成最小可运行的 question_texts / kc_texts
"""
import argparse
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.experiment import load_dataset_registry
from utils.preprocessing import (
    build_default_q_to_kc_mapping,
    collect_observed_question_ids,
    extract_kc_info,
    infer_question_ids,
    load_existing_q_to_kc_mapping,
    save_preprocessed_data,
    save_text_data,
)
from utils.project import project_path


def load_dataset_config(dataset_name, data_dir):
    """加载 data/datasets.toml 中的数据集配置。"""
    datasets = load_dataset_registry()
    if dataset_name not in datasets:
        raise ValueError(f"未知数据集: {dataset_name}")
    return datasets[dataset_name]


def main():
    parser = argparse.ArgumentParser(description="预处理 KT 数据以支持 GNN 和最小文本链路")
    parser.add_argument(
        "--dataset",
        type=str,
        default="assist09",
        choices=["assist09", "assist17", "statics", "algebra05", "doudouyun", "xes"],
        help="数据集名称",
    )
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="图结构输出目录")
    parser.add_argument("--text_output_dir", type=str, default="data/text_data", help="文本输出目录")
    parser.add_argument("--min_cooccurrence", type=int, default=5, help="最小共现次数阈值")
    parser.add_argument(
        "--force_text",
        action="store_true",
        help="即使 question_texts / kc_texts 已存在，也强制重写",
    )
    args = parser.parse_args()

    data_dir = project_path(args.data_dir)
    dataset_config = load_dataset_config(args.dataset, data_dir)
    inputs = list(dataset_config["inputs"])

    train_file = os.path.join(data_dir, dataset_config["train"])
    test_file = os.path.join(data_dir, dataset_config["test"]) if "test" in dataset_config else None
    question_text_path = project_path(args.text_output_dir, f"{args.dataset}_question_texts.json")
    kc_text_path = project_path(args.text_output_dir, f"{args.dataset}_kc_texts.json")

    print(f"加载数据集: {args.dataset}")
    print(f"训练文件: {train_file}")
    print(f"输入字段: {inputs}")

    question_id_mode = "from_existing_texts"
    q_to_kc = load_existing_q_to_kc_mapping(question_text_path)
    question_ids = None

    if q_to_kc is None:
        observed_q_ids = collect_observed_question_ids(
            [train_file, test_file] if test_file else [train_file],
            inputs,
        )
        expected_n_questions = int(dataset_config.get("n_questions", len(observed_q_ids)))
        question_ids, question_id_mode = infer_question_ids(observed_q_ids, expected_n_questions)
        q_to_kc, n_kc = build_default_q_to_kc_mapping(question_ids)
        print(f"推断题目空间: {question_id_mode}, q_count={len(question_ids)}, n_kc={n_kc}")
    else:
        question_ids = sorted(q_to_kc.keys())
        n_kc = max(q_to_kc.values()) + 1
        print(f"复用已有文本映射: q_count={len(question_ids)}, n_kc={n_kc}")

    if args.force_text or not (os.path.exists(question_text_path) and os.path.exists(kc_text_path)):
        save_text_data(
            args.dataset,
            question_ids,
            q_to_kc,
            n_kc,
            project_path(args.text_output_dir),
            project_path(args.data_dir),
        )
    else:
        print(f"文本数据已存在，跳过重写: {question_text_path}")

    kc_ids, edge_index, n_kc = extract_kc_info(
        train_file,
        inputs,
        q_to_kc,
        min_cooccurrence=args.min_cooccurrence,
    )

    save_preprocessed_data(
        kc_ids,
        edge_index,
        n_kc,
        project_path(args.output_dir),
        args.dataset,
        question_id_mode,
    )

    print("\n✅ 数据预处理完成！")
    print(f"知识点数量: {n_kc}")
    print(f"边数量: {edge_index.shape[1] if edge_index.ndim == 2 else 0}")
    print("\n使用方法:")
    print(f"  python DTransformer/preprocess_data.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()
