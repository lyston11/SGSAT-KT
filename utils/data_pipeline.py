import torch

from utils.embedding_artifacts import (
    load_edge_index,
    load_precomputed_embeddings,
    load_q_to_kc_mapping,
    load_text_data,
    resolve_precomputed_embedding_paths,
    validate_precomputed_embeddings,
)
from utils.kt_dataset import build_data_source, build_generated_valid_split


def prepare_bert_inputs(texts, q_ids, tokenizer, max_length=128):
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

    if not all_texts or all(text == "[PAD]" for text in all_texts):
        return None

    encoded = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def add_kc_ids_to_batch(q_batch, q_to_kc_mapping, n_kc=None):
    if isinstance(q_batch, list):
        return [add_kc_ids_to_batch(q, q_to_kc_mapping, n_kc) for q in q_batch]

    batch_size, seq_len = q_batch.shape
    kc_ids = torch.zeros_like(q_batch)

    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            q_id = q_batch[batch_idx, seq_idx].item()
            if q_id in q_to_kc_mapping:
                kc_id = q_to_kc_mapping[q_id]
                if n_kc is not None and kc_id >= n_kc:
                    kc_ids[batch_idx, seq_idx] = 0
                else:
                    kc_ids[batch_idx, seq_idx] = kc_id
            else:
                kc_ids[batch_idx, seq_idx] = 0

    return kc_ids
