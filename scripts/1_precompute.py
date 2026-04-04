#!/usr/bin/env python3
"""
阶段1: 预计算 Qwen 嵌入
一次性运行，生成 data/embeddings/*.pkl
"""
import argparse
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.experiment import load_yaml_config
from utils.precompute import (
    QwenEmbeddingGenerator,
    backfill_missing_kc_texts,
    load_precompute_text_assets,
    resolve_precompute_model_path,
)
from utils.project import project_path


def main():
    print("=" * 60)
    print("🔄 预计算 Qwen 嵌入")
    print("=" * 60)

    cfg = load_yaml_config("default.yaml")
    resolved_model_path = resolve_precompute_model_path(cfg)

    parser = argparse.ArgumentParser(description="TriSG-KT 预计算嵌入脚本")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=cfg.get("training", {}).get("dataset", "xes"),
        help="目标数据集名称，默认使用 configs/default.yaml 中的 training.dataset",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="预计算设备，默认 cuda；若无可用 GPU 会自动回退 cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="编码批大小，默认 32",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="文本编码最大 token 长度，默认 256；OOM 时会进一步自动下调",
    )
    args = parser.parse_args()

    print(f"📌 使用本地模型目录: {resolved_model_path}")
    generator = QwenEmbeddingGenerator(model_path=resolved_model_path, device=args.device)
    dataset_name = args.dataset

    print(f"📊 目标数据集: {dataset_name}")
    print("\n📂 加载文本数据...")

    text_assets = load_precompute_text_assets(dataset_name, project_path("data", "text_data"))
    q_file = text_assets["question_path"]
    kc_file = text_assets["kc_path"]
    questions = text_assets["questions"]
    kcs = text_assets["kcs"]

    if questions is not None:
        print(f"\n{'=' * 60}")
        print("📝 Step 1: 题目嵌入")
        print("=" * 60)
        generator.precompute_question_embeddings(
            questions,
            project_path("data", "embeddings", f"{dataset_name}_question_embeddings.pkl"),
            batch_size=args.batch_size,
            dataset_name=dataset_name,
            max_length=args.max_length,
        )
    else:
        print(f"⚠️  文件不存在: {q_file}")

    if kcs is not None:
        kcs, missing_kcs = backfill_missing_kc_texts(kcs, questions)
        if missing_kcs:
            print(f"⚠️  补全 {len(missing_kcs)} 个缺失 KC 文本: {missing_kcs}")

        print(f"\n{'=' * 60}")
        print("📚 Step 2: 知识点嵌入")
        print("=" * 60)
        generator.precompute_kc_embeddings(
            kcs,
            project_path("data", "embeddings", f"{dataset_name}_kc_embeddings.pkl"),
            batch_size=args.batch_size,
            dataset_name=dataset_name,
            max_length=args.max_length,
        )
    else:
        print(f"⚠️  文件不存在: {kc_file}")

    print(f"\n{'=' * 60}")
    print("🎉 预计算完成！")
    print("=" * 60)
    print(f"📁 嵌入文件: data/embeddings/{dataset_name}_question_embeddings.pkl")
    print(f"📁 嵌入文件: data/embeddings/{dataset_name}_kc_embeddings.pkl")
    print(f"💡 下一步: ./scripts/2_train.sh full --dataset {dataset_name}")


if __name__ == "__main__":
    main()
