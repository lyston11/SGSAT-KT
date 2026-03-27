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

# 阶段1: 检查预计算嵌入
echo "📂 检查预计算嵌入..."
if [ ! -f "data/embeddings/question_embeddings.pkl" ]; then
    echo "❌ 预计算嵌入不存在"
    echo ""
    echo "=========================================="
    echo "🔄 阶段1: 预计算 Qwen 嵌入"
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
else
    echo "✅ 预计算嵌入已存在，跳过"
fi

echo ""
echo "=========================================="
echo "🏋️  阶段2: 开始训练"
echo "=========================================="
echo "模式: $MODE"
echo ""

./scripts/2_train.sh $MODE

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
