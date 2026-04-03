#!/bin/bash
# 阶段1: 预计算Qwen嵌入

echo "🚀 开始预计算Qwen嵌入..."
echo "⏱️  预计时间: 15分钟"
echo ""

python scripts/1_precompute.py "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 预计算完成！"
    echo "📁 嵌入文件: data/embeddings/*.pkl"
    echo ""
    echo "下一步: 运行阶段2训练"
    echo "  ./scripts/2_train.sh full"
else
    echo ""
    echo "❌ 预计算失败"
    exit 1
fi
