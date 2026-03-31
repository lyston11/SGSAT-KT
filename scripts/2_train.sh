#!/bin/bash
# 阶段2: 训练模型

MODE=${1:-full}

# 创建日志目录
mkdir -p logs

# 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "🚀 开始训练..."
echo "模式: $MODE"
echo "📄 日志文件: $LOG_FILE"
echo ""

# 检查预计算嵌入（仅 LLM 相关模式需要）
if [ "$MODE" != "test" ] && [ "$MODE" != "baseline" ] && [ "$MODE" != "sakt" ] && [ "$MODE" != "akt" ] && [ "$MODE" != "dkt" ] && [ "$MODE" != "dkvmn" ]; then
    if [ ! -f "data/embeddings/question_embeddings.pkl" ]; then
        echo "⚠️  预计算嵌入不存在"
        echo "请先运行: ./scripts/1_precompute.sh"
        exit 1
    fi
fi

# 运行训练，同时输出到终端和日志文件
python scripts/2_train.py $MODE 2>&1 | tee "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ 训练完成！"
    echo "📄 日志已保存到: $LOG_FILE"
else
    echo ""
    echo "❌ 训练失败"
    echo "📄 查看日志: $LOG_FILE"
    exit 1
fi
