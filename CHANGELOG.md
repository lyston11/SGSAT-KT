# Changelog

All notable changes to SGSAT-KT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - 2026-03-14
- ✨ 根据研究方案文档精确实现三大修改点（100%符合文档要求）
- 📚 scripts目录完整重构和分类
  - 核心训练脚本: train.py, train_llm_gnn.py
  - 工具脚本: utils/ 目录
  - 可视化脚本: visualization/ 目录
  - 归档脚本: archive/ 目录
- 📖 创建完整的训练脚本说明文档 (scripts/README.md)
- 🔧 添加模型修改验证脚本 (scripts/utils/verify_modifications.py)
- 📝 更新SGSAT-GCER研究方案文档，添加实施状态部分
- 🎯 更新主README，反映最新的项目状态和脚本使用方法

### Changed - 2026-03-14
- 🔧 **修改点1（LLM语义Grounding）**: 改用直接相加 `id_emb + llm_emb` 而非加权平均
- 🔧 **修改点2（GNN先决图）**: 改用直接相加 `q_emb + prereq_emb` 而非加权平均
- 🔧 **修改点3（图增强DCF-Sim）**: 完整实现四个相似度分量，权重按文档要求
- 📁 重新组织scripts目录结构，提高可维护性
- 📚 更新所有文档以反映最新的实现状态

### Fixed - 2026-03-14
- 🐛 修复模型实现与文档要求不符的问题
- 🐛 清理过时的训练脚本，避免混淆
- 🐛 修正DCF-Sim相似度计算的hardcode值

### Added
- 项目结构重组和优化
- 完善的配置文件系统
- 工具函数模块 (utils/)
- Makefile 简化常用命令
- 更新文档和数据说明

### Changed
- 移动 BERT 模型到 `pretrained_models/` 目录
- 更新 .gitignore 文件
- 优化项目目录结构

### Fixed
- 修复数据加载路径问题
- 清理冗余文件

## [0.1.0] - 2025-03-06

### Added
- 三大创新点实现:
  - LLM语义Grounding
  - GNN Prerequisite Graph
  - Graph-Enhanced DCF-Sim
- 基线模型: DKT, AKT, DKVMN, SAKT
- 训练和评估脚本
- 数据集支持: assist09, assist17, algebra05, statics, doudouyun
- 完整的文档系统
