#!/bin/bash
# 项目结构检查和整理脚本
# 用途：快速检查项目文件组织情况，自动整理混乱的文件

echo "=================================="
echo "项目结构检查工具"
echo "=================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_structure() {
    echo "📂 检查标准目录..."
    for dir in docs logs scripts output tests utils; do
        if [ -d "$dir" ]; then
            echo -e "${GREEN}✓${NC} $dir/"
        else
            echo -e "${YELLOW}⚠${NC}  $dir/ (不存在)"
            read -p "  创建吗？ (y/n) " create
            if [ "$create" = "y" ]; then
                mkdir -p "$dir"
                echo -e "  ${GREEN}✓ 已创建 $dir/${NC}"
            fi
        fi
    done
    echo ""
}

check_root_files() {
    echo "📄 检查根目录文件..."
    echo ""

    # 检查需要移动的文件
    files_to_move=()

    # 检查文档文件
    for file in *.md *.txt; do
        if [ -f "$file" ]; then
            case "$file" in
                README.md|requirements.txt|pyproject.toml|Makefile|CHANGELOG.md|run_*.sh)
                    echo -e "${GREEN}✓${NC} $file (根目录保留)"
                    ;;
                *)
                    files_to_move+=("$file -> docs/")
                    echo -e "${YELLOW}⚠${NC}  $file (建议移到 docs/)"
                    ;;
            esac
        fi
    done

    # 检查脚本文件
    for file in *.py *.sh; do
        if [ -f "$file" ]; then
            case "$file" in
                run_*.sh)
                    echo -e "${GREEN}✓${NC} $file (根目录保留)"
                    ;;
                test_*.py)
                    files_to_move+=("$file -> tests/")
                    echo -e "${YELLOW}⚠${NC}  $file (建议移到 tests/)"
                    ;;
                monitor_*.sh|preprocess_*.py)
                    files_to_move+=("$file -> scripts/")
                    echo -e "${YELLOW}⚠${NC}  $file (建议移到 scripts/)"
                    ;;
                *)
                    if [[ "$file" =~ ^quick_|^temp_|^tmp_ ]]; then
                        echo -e "${RED}✗${NC} $file (临时文件，建议删除)"
                        files_to_delete+=("$file")
                    else
                        files_to_move+=("$file -> scripts/")
                        echo -e "${YELLOW}⚠${NC}  $file (建议移到 scripts/)"
                    fi
                    ;;
            esac
        fi
    done

    # 检查日志文件
    if ls *.log 1> /dev/null 2>&1; then
        for file in *.log; do
            files_to_move+=("$file -> logs/")
            echo -e "${RED}✗${NC} $file (应移到 logs/)"
        done
    fi

    echo ""
    if [ ${#files_to_move[@]} -gt 0 ]; then
        echo "发现 ${#files_to_move[@]} 个文件需要整理："
        for file in "${files_to_move[@]}"; do
            echo "  - $file"
        done
        echo ""
        read -p "是否自动整理这些文件？ (y/n) " cleanup
        if [ "$cleanup" = "y" ]; then
            cleanup_files
        fi
    else
        echo -e "${GREEN}✓ 根目录文件组织良好！${NC}"
    fi
    echo ""
}

cleanup_files() {
    echo ""
    echo "🧹 开始整理文件..."

    # 移动文档文件
    for file in *.md *.txt; do
        if [ -f "$file" ]; then
            case "$file" in
                README.md|requirements.txt|pyproject.toml|Makefile|CHANGELOG.md|run_*.sh)
                    continue
                    ;;
                *)
                    mv "$file" docs/ 2>/dev/null
                    echo -e "  ${GREEN}✓${NC} 移动 $file -> docs/"
                    ;;
            esac
        fi
    done

    # 移动脚本文件
    for file in *.py *.sh; do
        if [ -f "$file" ]; then
            case "$file" in
                run_*.sh)
                    continue
                    ;;
                test_*.py)
                    mkdir -p tests
                    mv "$file" tests/ 2>/dev/null
                    echo -e "  ${GREEN}✓${NC} 移动 $file -> tests/"
                    ;;
                *)
                    if [[ "$file" =~ ^quick_|^temp_|^tmp_ ]]; then
                        rm "$file" 2>/dev/null
                        echo -e "  ${GREEN}✓${NC} 删除临时文件 $file"
                    else
                        mv "$file" scripts/ 2>/dev/null
                        echo -e "  ${GREEN}✓${NC} 移动 $file -> scripts/"
                    fi
                    ;;
            esac
        fi
    done

    # 移动日志文件
    for file in *.log; do
        if [ -f "$file" ]; then
            mkdir -p logs
            mv "$file" logs/ 2>/dev/null
            echo -e "  ${GREEN}✓${NC} 移动 $file -> logs/"
        fi
    done

    echo ""
    echo -e "${GREEN}✓ 文件整理完成！${NC}"
    echo ""
}

check_log_organization() {
    echo "📋 检查日志文件组织..."
    echo ""

    if [ ! -d "logs" ]; then
        echo -e "${YELLOW}⚠${NC}  logs/ 目录不存在"
        return
    fi

    log_count=$(find logs -name "*.log" -type f 2>/dev/null | wc -l)
    echo "日志文件总数: $log_count"

    # 检查日志命名规范
    echo ""
    echo "检查日志命名规范..."
    for log in logs/*.log; do
        if [ -f "$log" ]; then
            basename=$(basename "$log")
            if [[ "$basename" =~ ^[a-z]+_[a-z]+_[a-z]+_[0-9]{8}_[0-9]{6}\.log$ ]]; then
                echo -e "${GREEN}✓${NC} $basename (规范)"
            else
                echo -e "${YELLOW}⚠${NC}  $basename (建议使用时间戳)"
            fi
        fi
    done

    # 显示最新日志
    echo ""
    echo "最新的5个日志文件："
    ls -lt logs/*.log 2>/dev/null | head -5 | awk '{print "  " $NF " (" $6 " " $7 " " $8 ")"}'
    echo ""
}

check_output_organization() {
    echo "💾 检查输出文件组织..."
    echo ""

    if [ ! -d "output" ]; then
        echo -e "${YELLOW}⚠${NC}  output/ 目录不存在"
        return
    fi

    # 检查子目录
    echo "输出目录结构："
    for dir in output/*/; do
        if [ -d "$dir" ]; then
            count=$(find "$dir" -name "*.pt" -o -name "*.pth" 2>/dev/null | wc -l)
            model_count=$(find "$dir" -name "model-*.pt" 2>/dev/null | wc -l)
            echo -e "  ${GREEN}✓${NC} ${dir} ($model_count 个模型文件)"
        fi
    done
    echo ""
}

show_project_score() {
    echo "=================================="
    echo "项目健康度评分"
    echo "=================================="
    echo ""

    score=100
    issues=()

    # 检查根目录文件
    non_standard_files=$(find . -maxdepth 1 -type f \( -name "*.md" -o -name "*.txt" -o -name "*.py" -o -name "*.sh" \) ! -name "README.md" ! -name "requirements.txt" ! -name "pyproject.toml" ! -name "Makefile" ! -name "CHANGELOG.md" ! -name "run_*.sh" | wc -l)
    if [ "$non_standard_files" -gt 0 ]; then
        issues+=("根目录有 $non_standard_files 个非标准文件 (-10分)")
        score=$((score - 10))
    fi

    # 检查日志
    if [ ! -d "logs" ]; then
        issues+=("缺少 logs/ 目录 (-20分)")
        score=$((score - 20))
    fi

    # 检查文档
    if [ ! -d "docs" ]; then
        issues+=("缺少 docs/ 目录 (-20分)")
        score=$((score - 20))
    fi

    # 检查脚本
    if [ ! -d "scripts" ]; then
        issues+=("缺少 scripts/ 目录 (-20分)")
        score=$((score - 20))
    fi

    # 检查输出
    if [ ! -d "output" ]; then
        issues+=("缺少 output/ 目录 (-20分)")
        score=$((score - 20))
    fi

    # 检查项目结构文档
    if [ ! -f "docs/PROJECT_STRUCTURE.md" ]; then
        issues+=("缺少 PROJECT_STRUCTURE.md (-10分)")
        score=$((score - 10))
    fi

    # 显示评分
    if [ $score -eq 100 ]; then
        echo -e "${GREEN}满分：100分！ 🎉${NC}"
        echo "项目结构完美，继续保持！"
    elif [ $score -ge 80 ]; then
        echo -e "${GREEN}得分：$score/100 分${NC}"
        echo "项目结构良好，有小幅改进空间"
    elif [ $score -ge 60 ]; then
        echo -e "${YELLOW}得分：$score/100 分${NC}"
        echo "项目结构尚可，建议整理"
    else
        echo -e "${RED}得分：$score/100 分${NC}"
        echo "项目结构混乱，急需整理"
    fi

    echo ""
    if [ ${#issues[@]} -gt 0 ]; then
        echo "发现的问题："
        for issue in "${issues[@]}"; do
            echo -e "  ${YELLOW}⚠${NC} $issue"
        done
        echo ""
    fi
}

# 主函数
main() {
    check_structure
    check_root_files
    check_log_organization
    check_output_organization
    show_project_score

    echo "=================================="
    echo "检查完成！"
    echo "=================================="
}

# 运行主函数
main
