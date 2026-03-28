#!/bin/bash
# 完整训练脚本：预计算 + 训练
# 第一次使用时自动运行两阶段

set -e

MODE=${1:-full}

echo "=========================================="
echo "🚀 SGSAT-KT 完整训练流程"
echo "=========================================="
echo "模式: $MODE"
echo ""

# 阶段1: 检查预计算嵌入与配置是否匹配
echo "📂 检查预计算嵌入..."
NEED_PRECOMPUTE=false

if [ ! -f "data/embeddings/question_embeddings.pkl" ]; then
    echo "❌ 预计算嵌入不存在"
    NEED_PRECOMPUTE=true
else
    # 检查嵌入维度是否与当前配置的模型匹配
    EMBED_DIM=$(python -c "
import pickle, yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
model_path = cfg.get('llm', {}).get('pretrained_model', 'pretrained_models/bert-base-chinese')
with open('data/embeddings/question_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'{data.get(\"hidden_size\", \"?\")}|{model_path}')
" 2>/dev/null)
    EMBED_HIDDEN=$(echo "$EMBED_DIM" | cut -d'|' -f1)
    EMBED_MODEL=$(echo "$EMBED_DIM" | cut -d'|' -f2)
    echo "📊 当前嵌入: 模型=$EMBED_MODEL, 维度=$EMBED_HIDDEN"
    echo "📄 配置模型: $(grep 'pretrained_model' configs/default.yaml | head -1 | awk '{print $2}' | tr -d '\"')"

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
    else
        echo "✅ 预计算嵌入与配置匹配，跳过"
    fi
fi

if [ "$NEED_PRECOMPUTE" = true ]; then
    echo ""
    echo "=========================================="
    echo "🔄 阶段1: 预计算嵌入"
    echo "=========================================="
    echo "⏱️  预计时间: 15分钟"
    echo ""

    ./scripts/1_precompute.sh

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
echo ""

./scripts/2_train.sh $MODE  # 从项目根目录运行

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
