#!/bin/bash
# 完整训练脚本：预计算 + 训练
# 第一次使用时自动运行两阶段

set -e

MODE=${1:-full}
CURRENT_DATASET=${2:-}

if [ -z "$CURRENT_DATASET" ]; then
    CURRENT_DATASET=$(python - <<'PY'
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('training', {}).get('dataset', 'xes'))
PY
)
fi

echo "=========================================="
echo "🚀 TriSG-KT 完整训练流程"
echo "=========================================="
echo "模式: $MODE"
echo "数据集: $CURRENT_DATASET"
echo ""

# 阶段1: 检查预计算嵌入（仅 LLM 相关模式需要）
NO_LLM_MODES="test baseline sakt akt dkt dkvmn"
NEED_PRECOMPUTE=false

if echo " $NO_LLM_MODES " | grep -q " $MODE "; then
    echo "📂 模式 $MODE 不需要预计算嵌入，跳过"
else
    echo "📂 检查预计算嵌入..."

    QUESTION_EMB_PATH="data/embeddings/${CURRENT_DATASET}_question_embeddings.pkl"
    KC_EMB_PATH="data/embeddings/${CURRENT_DATASET}_kc_embeddings.pkl"
    if [ ! -f "$QUESTION_EMB_PATH" ] && [ -f "data/embeddings/question_embeddings.pkl" ]; then
        QUESTION_EMB_PATH="data/embeddings/question_embeddings.pkl"
    fi
    if [ ! -f "$KC_EMB_PATH" ] && [ -f "data/embeddings/kc_embeddings.pkl" ]; then
        KC_EMB_PATH="data/embeddings/kc_embeddings.pkl"
    fi

    if [ ! -f "$QUESTION_EMB_PATH" ]; then
        echo "❌ 预计算嵌入不存在"
        NEED_PRECOMPUTE=true
    elif [ ! -f "$KC_EMB_PATH" ]; then
        echo "❌ 知识点预计算嵌入不存在"
        NEED_PRECOMPUTE=true
    else
        # 检查嵌入维度、数据集元信息和 KC 覆盖
        EMBED_INFO=$(CURRENT_DATASET="$CURRENT_DATASET" QUESTION_EMB_PATH="$QUESTION_EMB_PATH" KC_EMB_PATH="$KC_EMB_PATH" python - <<'PY'
import json
import os
import pickle
import yaml
from pathlib import Path

with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)

dataset = os.environ.get('CURRENT_DATASET') or cfg.get('training', {}).get('dataset', 'xes')
model_path = cfg.get('llm', {}).get('pretrained_model', 'pretrained_models/bert-base-chinese')

with open(os.environ['QUESTION_EMB_PATH'], 'rb') as f:
    q_data = pickle.load(f)
with open(os.environ['KC_EMB_PATH'], 'rb') as f:
    kc_data = pickle.load(f)

required_kc = set()
expected_q_count = None
text_path = Path('data/text_data') / f'{dataset}_question_texts.json'
if text_path.exists():
    with text_path.open(encoding='utf-8') as f:
        q_texts = json.load(f)
    expected_q_count = len(q_texts)
    for info in q_texts.values():
        skill = info.get('skill', '-1')
        try:
            kc_id = int(skill)
        except Exception:
            continue
        if kc_id >= 0:
            required_kc.add(kc_id)

kc_ids = {int(x) for x in kc_data.get('kc_ids', [])}
missing_kc = sorted(required_kc - kc_ids)

print(
    "|".join(
        [
            str(q_data.get('hidden_size', '?')),
            str(q_data.get('model_path', '')),
            str(model_path),
            str(q_data.get('dataset_name', '')),
            str(kc_data.get('dataset_name', '')),
            str(len(q_data.get('question_ids', []))),
            str(expected_q_count if expected_q_count is not None else ''),
            str(len(missing_kc)),
        ]
    )
)
PY
)
        EMBED_HIDDEN=$(echo "$EMBED_INFO" | cut -d'|' -f1)
        EMBED_MODEL=$(echo "$EMBED_INFO" | cut -d'|' -f2)
        CONFIG_MODEL=$(echo "$EMBED_INFO" | cut -d'|' -f3)
        EMBED_Q_DATASET=$(echo "$EMBED_INFO" | cut -d'|' -f4)
        EMBED_KC_DATASET=$(echo "$EMBED_INFO" | cut -d'|' -f5)
        EMBED_Q_COUNT=$(echo "$EMBED_INFO" | cut -d'|' -f6)
        EXPECTED_Q_COUNT=$(echo "$EMBED_INFO" | cut -d'|' -f7)
        MISSING_KC_COUNT=$(echo "$EMBED_INFO" | cut -d'|' -f8)
        echo "📊 当前嵌入: 模型=$EMBED_MODEL, 维度=$EMBED_HIDDEN, q_dataset=${EMBED_Q_DATASET:-<missing>}, kc_dataset=${EMBED_KC_DATASET:-<missing>}, q_count=$EMBED_Q_COUNT"
        echo "📄 配置模型: $CONFIG_MODEL"

        # 通过实际加载模型获取真实维度来校验
        EXPECTED_DIM=$(python -c "
import json, os, yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
model_path = cfg.get('llm', {}).get('pretrained_model', 'pretrained_models/bert-base-chinese')
cfg_path = os.path.join(model_path, 'config.json')
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        mc = json.load(f)
    print(mc.get('hidden_size', 'unknown'))
else:
    print('unknown')
" 2>/dev/null)

        if [ "$EMBED_HIDDEN" != "$EXPECTED_DIM" ]; then
            echo "⚠️  嵌入维度($EMBED_HIDDEN)与模型期望维度($EXPECTED_DIM)不匹配"
            NEED_PRECOMPUTE=true
        elif [ -z "$EMBED_Q_DATASET" ] || [ -z "$EMBED_KC_DATASET" ]; then
            echo "⚠️  预计算嵌入缺少 dataset_name 元信息，需要重新生成"
            NEED_PRECOMPUTE=true
        elif [ "$EMBED_Q_DATASET" != "$EMBED_KC_DATASET" ]; then
            echo "⚠️  题目嵌入与知识点嵌入来自不同数据集"
            NEED_PRECOMPUTE=true
        elif [ "$EMBED_Q_DATASET" != "$CURRENT_DATASET" ]; then
            echo "⚠️  预计算嵌入数据集与当前配置不一致"
            NEED_PRECOMPUTE=true
        elif [ -n "$EXPECTED_Q_COUNT" ] && [ "$EMBED_Q_COUNT" != "$EXPECTED_Q_COUNT" ]; then
            echo "⚠️  题目嵌入数量($EMBED_Q_COUNT)与当前文本数据期望数量($EXPECTED_Q_COUNT)不一致"
            NEED_PRECOMPUTE=true
        elif [ "$MISSING_KC_COUNT" != "0" ]; then
            echo "⚠️  检测到 $MISSING_KC_COUNT 个被题目引用的 KC 缺少预计算嵌入"
            NEED_PRECOMPUTE=true
        else
            echo "✅ 预计算嵌入与配置匹配，跳过"
        fi
    fi
fi

if [ "$NEED_PRECOMPUTE" = true ]; then
    echo ""
    echo "=========================================="
    echo "🔄 阶段1: 预计算嵌入"
    echo "=========================================="
    echo "⏱️  预计时间: 15分钟"
    echo ""

    ./scripts/1_precompute.sh "$CURRENT_DATASET"

    if [ $? -ne 0 ]; then
        echo "❌ 预计算失败"
        exit 1
    fi

    echo ""
    echo "✅ 预计算完成！"
fi

echo ""
echo "=========================================="
echo "🏋️  阶段2: 开始训练"
echo "=========================================="
echo "模式: $MODE"
echo "数据集: $CURRENT_DATASET"
echo ""

./scripts/2_train.sh "$MODE" --dataset "$CURRENT_DATASET"  # 从项目根目录运行

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "🎉 训练完成！"
    echo "=========================================="
else
    echo ""
    echo "❌ 训练失败"
    exit 1
fi
