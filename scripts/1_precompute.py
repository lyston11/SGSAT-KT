#!/usr/bin/env python3
"""
阶段1: 预计算 Qwen 嵌入
一次性运行，生成 data/embeddings/*.pkl
"""
import os
import json
import pickle
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, project_root)

from scripts.utils.config_loader import ConfigLoader


class QwenEmbeddingGenerator:
    """Qwen 语义嵌入生成器 (使用sentence-transformers)"""

    def __init__(self, model_path="pretrained_models/qwen3-4b", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        print(f"📦 加载 Qwen 模型: {model_path}")
        print(f"🔧 设备: {self.device}")

        # 使用SentenceTransformer加载；若是本地目录则强制本地读取，避免误走Hub
        local_only = os.path.isdir(model_path)
        model_kwargs = {"trust_remote_code": True}
        if local_only:
            model_kwargs["local_files_only"] = True

        self.model = SentenceTransformer(
            model_path,
            device=self.device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"local_files_only": local_only},
        )
        self.hidden_size = self.model.get_sentence_embedding_dimension()
        print(f"✅ 模型加载完成，嵌入维度: {self.hidden_size}")

    def encode_text(self, text, max_length=512):
        """编码单条文本"""
        # SentenceTransformer自动处理tokenization
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding.flatten()

    def batch_encode_texts(self, texts, batch_size=32, max_length=512, desc="编码文本"):
        """批量编码文本"""
        # 使用SentenceTransformer的批量编码
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embeddings

    def precompute_question_embeddings(self, questions_data, output_path, batch_size=32):
        """预计算题目嵌入"""
        print(f"📝 预计算 {len(questions_data)} 个题目嵌入...")

        question_ids = sorted(questions_data.keys())
        question_texts = [questions_data[qid].get("text") or questions_data[qid].get("content", "") for qid in question_ids]

        question_embeddings = self.batch_encode_texts(
            question_texts,
            batch_size=batch_size,
            desc="编码题目文本"
        )

        result = {
            "question_ids": question_ids,
            "embeddings": question_embeddings,
            "hidden_size": self.hidden_size,
            "model_path": self.model_path,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)

        print(f"✅ 题目嵌入已保存: {output_path}")
        return result

    def precompute_kc_embeddings(self, kc_data, output_path, batch_size=32):
        """预计算知识点嵌入"""
        print(f"📚 预计算 {len(kc_data)} 个知识点嵌入...")

        kc_ids = sorted(kc_data.keys())
        kc_texts = [kc_data[kid] for kid in kc_ids]

        kc_embeddings = self.batch_encode_texts(
            kc_texts,
            batch_size=batch_size,
            desc="编码知识点文本"
        )

        result = {
            "kc_ids": kc_ids,
            "embeddings": kc_embeddings,
            "hidden_size": self.hidden_size,
            "model_path": self.model_path,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)

        print(f"✅ 知识点嵌入已保存: {output_path}")
        return result


def main():
    print("=" * 60)
    print("🔄 预计算 Qwen 嵌入")
    print("=" * 60)

    # 从统一配置读取 llm.pretrained_model，并解析为本地路径优先
    loader = ConfigLoader()
    cfg = loader.load_yaml('default.yaml')
    model_path_cfg = (
        cfg.get('llm', {}).get('pretrained_model')
        or 'pretrained_models/qwen3-4b'
    )

    if os.path.isabs(model_path_cfg):
        resolved_model_path = model_path_cfg
    else:
        resolved_model_path = os.path.join(project_root, model_path_cfg)

    # 兼容旧目录名
    legacy_model_path = os.path.join(project_root, 'pretrained_models/Qwen3-Embedding-4B')
    if not os.path.isdir(resolved_model_path) and os.path.isdir(legacy_model_path):
        print(f"⚠️  配置路径不存在，回退使用旧目录: {legacy_model_path}")
        resolved_model_path = legacy_model_path

    if not os.path.isdir(resolved_model_path):
        raise FileNotFoundError(
            f"本地模型目录不存在: {resolved_model_path}\n"
            f"请检查 configs/default.yaml 的 llm.pretrained_model，"
            f"当前建议为 pretrained_models/qwen3-4b"
        )

    print(f"📌 使用本地模型目录: {resolved_model_path}")
    generator = QwenEmbeddingGenerator(model_path=resolved_model_path)

    # 加载文本数据
    print("\n📂 加载文本数据...")
    data_dir = os.path.join(project_root, "data/text_data")

    q_file = os.path.join(data_dir, "xes_question_texts.json")
    kc_file = os.path.join(data_dir, "xes_kc_texts.json")

    if os.path.exists(q_file):
        with open(q_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        print(f"\n" + "=" * 60)
        print("📝 Step 1: 题目嵌入")
        print("=" * 60)
        generator.precompute_question_embeddings(
            questions,
            os.path.join(project_root, "data/embeddings/question_embeddings.pkl")
        )
    else:
        print(f"⚠️  文件不存在: {q_file}")

    if os.path.exists(kc_file):
        with open(kc_file, 'r', encoding='utf-8') as f:
            kcs = json.load(f)

        print(f"\n" + "=" * 60)
        print("📚 Step 2: 知识点嵌入")
        print("=" * 60)
        generator.precompute_kc_embeddings(
            kcs,
            os.path.join(project_root, "data/embeddings/kc_embeddings.pkl")
        )
    else:
        print(f"⚠️  文件不存在: {kc_file}")

    print("\n" + "=" * 60)
    print("🎉 预计算完成！")
    print("=" * 60)
    print("📁 嵌入文件: data/embeddings/")
    print("💡 下一步: ./scripts/2_train.sh full")


if __name__ == "__main__":
    main()
