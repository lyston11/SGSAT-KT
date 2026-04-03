#!/usr/bin/env python3
"""
阶段2: 训练模型
使用预计算嵌入进行高效训练
"""
import os
import sys

import torch

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.data_pipeline import (
    build_data_source as pipeline_build_data_source,
    build_generated_valid_split as pipeline_build_generated_valid_split,
    load_edge_index as pipeline_load_edge_index,
    load_precomputed_embeddings as pipeline_load_precomputed_embeddings,
    load_q_to_kc_mapping as pipeline_load_q_to_kc_mapping,
    load_text_data as pipeline_load_text_data,
    resolve_precomputed_embedding_paths as pipeline_resolve_precomputed_embedding_paths,
    validate_precomputed_embeddings as pipeline_validate_precomputed_embeddings,
)
from utils.experiment import (
    flatten_config as flatten_runtime_config,
    load_dataset_registry,
    load_mode_config,
)
from utils.training import (
    build_optimizer_and_scheduler,
    create_output_dir,
    initialize_runtime,
    load_best_model_if_available,
    save_best_model,
    save_metrics_history,
    save_training_summary,
    train_epoch,
    validate,
)

from DTransformer.model import DTransformer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TriSG-KT 训练脚本")
    parser.add_argument('mode', choices=['test', 'baseline', 'full', 'prod', 'sakt', 'akt', 'dkt', 'dkvmn'], help='训练模式')
    parser.add_argument('--device', help='覆盖设备设置')
    parser.add_argument('--dataset', help='覆盖数据集设置')
    args = parser.parse_args()

    # 统一从 default.yaml 加载配置
    try:
        config = load_mode_config(
            args.mode,
            dataset=args.dataset,
            device=args.device,
        )

        print(f"✅ 从 configs/default.yaml 加载配置")
        print(f"📝 使用预设: {args.mode}")

    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 扁平化配置，方便后续访问
    config = flatten_runtime_config(config)

    # 打印配置
    print("=" * 60)
    print(f"🚀 训练模式: {args.mode}")
    print("=" * 60)
    # 打印主要配置
    print(f"  批大小: {config.get('training_batch_size', config.get('batch_size', 'N/A'))}")
    print(f"  学习率: {config.get('training_learning_rate', config.get('learning_rate', 'N/A'))}")
    print(f"  训练轮数: {config.get('training_n_epochs', config.get('epochs', 'N/A'))}")
    print(f"  使用LLM: {config.get('llm_use_llm', config.get('use_llm', 'N/A'))}")
    print(f"  使用GNN: {config.get('gnn_use_gnn', config.get('use_gnn', 'N/A'))}")
    print(f"  GPU设备: {config.get('gpu_device_ids', 'N/A')}")
    print("=" * 60)

    # 加载数据
    print("\n📂 加载数据...")
    datasets = load_dataset_registry()
    dataset_name = config.get('training_dataset', config.get('dataset', 'xes'))
    dataset_config = datasets[dataset_name]

    # 检查预计算嵌入（仅 use_llm=True 时有意义）
    if config.get('use_precomputed', False) and config.get('use_llm', False):
        precomputed_paths = pipeline_resolve_precomputed_embedding_paths(
            os.path.join(project_root, 'data'),
            dataset_name,
        )
        if not os.path.exists(precomputed_paths["question"]):
            print("\n❌ 预计算嵌入不存在！")
            print(f"请先运行: ./scripts/1_precompute.sh {dataset_name}")
            sys.exit(1)
        print(f"✅ 预计算嵌入检查通过: {os.path.basename(precomputed_paths['question'])}")

    data_dir = os.path.join(project_root, "data")
    seq_len = dataset_config.get("seq_len", None)
    config['seq_len'] = seq_len  # 添加到 config 中

    batch_size = config.get('batch_size', config.get('training_batch_size', 16))
    test_batch_size = config.get('test_batch_size', config.get('training_test_batch_size', 8))
    train_path = os.path.join(data_dir, dataset_config["train"])
    split_info = None

    if "valid" in dataset_config:
        train_data = pipeline_build_data_source(
            train_path,
            dataset_config["inputs"],
            batch_size=batch_size,
            seq_len=seq_len,
            shuffle=True,
        )
        valid_data = pipeline_build_data_source(
            os.path.join(data_dir, dataset_config["valid"]),
            dataset_config["inputs"],
            batch_size=test_batch_size,
            seq_len=seq_len,
            shuffle=False,
        )
        split_info = {
            "source": "provided_files",
            "train_path": dataset_config["train"],
            "valid_path": dataset_config["valid"],
        }
    else:
        valid_ratio = float(config.get('training_validation_ratio', config.get('validation_ratio', 0.1)))
        valid_seed = int(config.get('training_validation_seed', config.get('validation_seed', 42)))
        train_data, valid_data, split_info = pipeline_build_generated_valid_split(
            train_path,
            dataset_config["inputs"],
            seq_len,
            batch_size,
            test_batch_size,
            valid_ratio=valid_ratio,
            seed=valid_seed,
        )
        print(
            f"⚠️  数据集未提供 valid 划分，已从 train.txt 中生成验证集 "
            f"(ratio={valid_ratio:.2f}, seed={valid_seed})"
        )

    # 独立测试集（训练中不使用，仅在训练结束后做最终评估）
    test_data = None
    if "test" in dataset_config:
        test_data = pipeline_build_data_source(
            os.path.join(data_dir, dataset_config["test"]),
            dataset_config["inputs"],
            batch_size=test_batch_size,
            seq_len=seq_len,
            shuffle=False,
        )

    config['validation_split_info'] = split_info
    if test_data is not None:
        print(f"✅ 数据加载完成: {dataset_name} (train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)})")
    else:
        print(f"✅ 数据加载完成: {dataset_name} (train={len(train_data)}, valid={len(valid_data)})")

    # 加载文本数据（如果使用LLM）
    q_texts = None
    tokenizer = None
    if config.get('use_llm', False) and not config.get('use_precomputed', False):
        q_texts = pipeline_load_text_data(dataset_name, data_dir)
        if q_texts is None:
            print("⚠️  无法加载文本数据，关闭LLM")
            config['use_llm'] = False
        else:
            from transformers import AutoTokenizer
            model_path = config.get('pretrained_model', 'bert-base-chinese')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"✓ 加载tokenizer: {model_path}")

    # 加载q->kc映射（GNN 或 LLM 对比损失都需要）
    q_to_kc_mapping = None
    edge_index = None
    need_kc_mapping = config.get('use_gnn', False) or config.get('use_llm', False)
    if need_kc_mapping:
        q_to_kc_mapping = pipeline_load_q_to_kc_mapping(dataset_name, data_dir)
        if q_to_kc_mapping is None:
            print("⚠️  无法加载q->kc映射")
            if config.get('use_gnn', False):
                print("   关闭GNN")
                config['use_gnn'] = False
        else:
            # 自动校正 n_kc，避免配置值过小导致索引越界
            inferred_n_kc = max(q_to_kc_mapping.values()) + 1 if q_to_kc_mapping else 0
            configured_n_kc = config.get('gnn_n_kc', config.get('n_kc', 100))
            if inferred_n_kc > configured_n_kc:
                print(f"⚠️  检测到 n_kc 配置过小({configured_n_kc})，自动提升到 {inferred_n_kc}")
                config['n_kc'] = inferred_n_kc
                config['gnn_n_kc'] = inferred_n_kc

    # edge_index 仅 GNN 需要
    if config.get('use_gnn', False) and q_to_kc_mapping is not None:
        edge_index, edge_max_kc = pipeline_load_edge_index(dataset_name, data_dir, torch.device('cpu'))
        if edge_index is None:
            print("⚠️  未找到edge_index，GNN先决图分支将被跳过")
        elif edge_max_kc is not None:
            current_n_kc = config.get('gnn_n_kc', config.get('n_kc', 100))
            if edge_max_kc >= current_n_kc:
                required_n_kc = edge_max_kc + 1
                print(f"⚠️  edge_index 包含 kc_id={edge_max_kc}，超出 n_kc={current_n_kc}")
                print(f"🔧 自动提升 n_kc: {current_n_kc} -> {required_n_kc}")
                config['n_kc'] = required_n_kc
                config['gnn_n_kc'] = required_n_kc

    # 加载预计算嵌入（仅在 LLM 分支启用时有意义）
    precomputed_embeddings = None
    if config.get('use_precomputed', False) and config.get('use_llm', False):
        try:
            precomputed_embeddings = pipeline_load_precomputed_embeddings(
                data_dir,
                dataset_name,
                use_gnn=config.get('use_gnn', False),
                use_llm=True,
            )
            embedding_issues = pipeline_validate_precomputed_embeddings(
                precomputed_embeddings,
                dataset_name,
                dataset_config,
                q_to_kc_mapping=q_to_kc_mapping,
            )
            if embedding_issues:
                issues_text = "\n".join(f"  - {issue}" for issue in embedding_issues)
                raise RuntimeError(
                    f"预计算嵌入校验失败，请重新运行 ./scripts/1_precompute.sh {dataset_name}：\n"
                    f"{issues_text}"
                )
            print("✅ 预计算嵌入已加载到训练流程")
        except Exception as e:
            print(f"❌ 预计算嵌入不可用: {e}")
            sys.exit(1)
    elif config.get('use_precomputed', False) and not config.get('use_llm', False):
        print("ℹ️  use_llm=False，跳过预计算嵌入加载")
        config['use_precomputed'] = False

    # 预计算回退：若关闭了预计算且尚未准备在线文本输入，则补充加载
    if config.get('use_llm', False) and not config.get('use_precomputed', False) and q_texts is None:
        q_texts = pipeline_load_text_data(dataset_name, data_dir)
        if q_texts is None:
            print("⚠️  回退在线文本分支失败：文本数据缺失，LLM分支将退化为ID嵌入")
        else:
            from transformers import AutoTokenizer
            model_path = config.get('pretrained_model', 'bert-base-chinese')
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"✓ 回退成功，已加载在线tokenizer: {model_path}")

    if config.get('use_graph_similarity', config.get('recommendation_use_graph_similarity', False)):
        print("ℹ️  use_graph_similarity=true: DCFSimGraphEnhanced 是后处理工具类，"
              "用于训练后的用户相似度分析，不影响训练过程")

    # 创建模型
    print("\n🤖 创建模型...")
    baseline_name = config.get('model__baseline', None)
    if baseline_name:
        from baselines import create_baseline_model, BaselineWrapper
        raw_model = create_baseline_model(
            baseline_name,
            dataset_config["n_questions"],
            d_model=config.get('model_d_model', config.get('d_model', 256)),
            n_heads=config.get('model_n_heads', config.get('n_heads', 8)),
            dropout=config.get('model_dropout', config.get('dropout', 0.2)),
            batch_size=config.get('batch_size', config.get('training_batch_size', 16)),
            device='cpu',
        )
        model = BaselineWrapper(raw_model)
        print(f"✅ 基线模型创建完成: {baseline_name}")
    else:
        # 确保 n_kc 配置正确
        final_n_kc = config.get('gnn_n_kc', config.get('n_kc', 100))
        print(f"📊 使用 n_kc={final_n_kc} 创建模型")
        if edge_index is not None:
            print(f"📊 edge_index max_kc={edge_index.max().item()}")
        model = DTransformer(
            dataset_config["n_questions"],
            dataset_config.get("n_pid", 0),
            d_model=config.get('model_d_model', config.get('d_model', 256)),
            d_fc=config.get('model_d_fc', config.get('d_fc', 512)),
            n_heads=config.get('model_n_heads', config.get('n_heads', 8)),
            n_layers=config.get('model_n_layers', config.get('n_layers', 2)),
            n_know=config.get('model_n_know', config.get('n_know', 64)),
            lambda_cl=config.get('recommendation_lambda_cl', config.get('lambda_cl', 0.1)),
            dropout=config.get('model_dropout', config.get('dropout', 0.2)),
            proj=config.get('proj', False),
            hard_neg=config.get('hard_neg', False),
            window=config.get('model_window', config.get('window', 1)),
            use_llm=config.get('use_llm', False),
            pretrained_model=config.get('pretrained_model', 'bert-base-chinese'),
            precomputed_embeddings=precomputed_embeddings,
            id_dim=config.get('model_id_dim', config.get('id_dim', 128)),
            llm_proj_dim=config.get('model_llm_proj_dim', config.get('llm_proj_dim', 256)),
            llm_inter_dim=config.get('model_llm_inter_dim', config.get('llm_inter_dim', 512)),
            id_dropout_rate=config.get('model_id_dropout_rate', config.get('id_dropout_rate', 0.15)),
            lambda_contra=config.get('llm_lambda_contra', config.get('lambda_contra', 0.3)),
            contrast_temperature=config.get('llm_contrast_temperature', config.get('contrast_temperature', 0.07)),
            use_gnn=config.get('use_gnn', False),
            cross_attn_heads=config.get('model_cross_attn_heads', 4),
            freeze_bert=config.get('llm_freeze_bert', True),
            n_kc=final_n_kc,
            gnn_layers=config.get('gnn_gnn_layers', config.get('gnn_layers', 2)),
        )
        print(f"✅ 模型创建完成，实际 n_kc={model.n_kc}")

    # 分支激活报告：避免配置开启但链路未生效
    llm_enabled = bool(config.get('use_llm', False))
    precomputed_active = bool(config.get('use_precomputed', False) and precomputed_embeddings is not None)
    online_text_active = bool(llm_enabled and q_texts is not None and tokenizer is not None)
    gnn_enabled = bool(config.get('use_gnn', False))
    gnn_active = bool(gnn_enabled and edge_index is not None)
    cl_loss_active = bool(config.get('cl_loss', config.get('recommendation_cl_loss', False)))

    print("📌 分支激活状态:")
    print(f"  use_llm: {llm_enabled}")
    print(f"  use_precomputed: {precomputed_active}")
    print(f"  use_online_text: {online_text_active}")
    print(f"  use_gnn: {gnn_active}")
    print(f"  use_cl_loss: {cl_loss_active}")

    if gnn_enabled and edge_index is not None:
        print(f"  gnn_edges: {edge_index.size(1)}")

    print("📌 分支诊断:")
    if not llm_enabled:
        print("  LLM分支未启用：当前配置 use_llm=False（test/baseline 预设默认关闭）")
    else:
        if not precomputed_active and not online_text_active:
            print("  LLM分支退化为ID：预计算未加载且在线文本未就绪")
        elif precomputed_active:
            print("  LLM分支生效：使用预计算嵌入")
        elif online_text_active:
            print("  LLM分支生效：使用在线文本编码")

    if not gnn_enabled:
        print("  GNN分支未启用：当前配置 use_gnn=False（test/baseline 预设默认关闭）")
    elif edge_index is None:
        print("  GNN分支未生效：未找到 edge_index 文件")
    else:
        print("  GNN分支生效：edge_index 已加载")

    if not cl_loss_active:
        print("  CL损失未启用：recommendation.cl_loss=False")
    else:
        print("  CL损失已启用：训练将调用 get_cl_loss")

    # 优化器 + Cosine Annealing 调度器
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    device, edge_index, scaler, use_amp = initialize_runtime(
        model,
        config,
        baseline_name=baseline_name,
        edge_index=edge_index,
    )

    # 创建输出目录
    output_dir = create_output_dir(project_root, args.mode, dataset_name, config, split_info)

    # 训练循环
    epochs = config.get('epochs', config.get('n_epochs', 30))
    print(f"\n🏋️  开始训练 {epochs} epochs...")
    best = {"auc": 0}
    best_epoch = 0
    history = []
    history_path = os.path.join(output_dir, "metrics_history.json") if output_dir else None
    summary_path = os.path.join(output_dir, "summary.json") if output_dir else None
    test_results = None

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")

        # 训练
        avg_loss = train_epoch(
            model,
            train_data,
            optimizer,
            config,
            q_texts,
            tokenizer,
            q_to_kc_mapping,
            edge_index,
            scaler,
            use_amp,
        )
        print(f"训练损失: {avg_loss:.4f}")

        # 验证
        print("验证中...")
        results = validate(
            model,
            valid_data,
            config,
            q_texts,
            tokenizer,
            q_to_kc_mapping,
            edge_index,
            use_amp,
        )
        print(f"验证结果: {results}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "lr": optimizer.param_groups[0]['lr'],
            **results,
        }
        history.append(epoch_record)
        save_metrics_history(history_path, history)

        # 保存最佳模型
        if results["auc"] > best["auc"]:
            best = results
            best_epoch = epoch
            save_best_model(model, output_dir)

        # 早停
        early_stop = config.get('training_early_stop', config.get('early_stop', 0))
        if early_stop > 0 and epoch - best_epoch >= early_stop:
            print(f"⏸️  早停触发 (Best: Epoch {best_epoch})")
            break

        # Cosine Annealing: 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📐 学习率: {current_lr:.6f}")

    print(f"\n{'='*60}")
    print(f"🎉 训练完成!")
    print(f"{'='*60}")
    print(f"最佳验证结果 (Epoch {best_epoch}):")
    for k, v in best.items():
        print(f"  {k}: {v:.4f}")

    # 最终测试评估（仅在训练结束后跑一次，不影响模型选择）
    if test_data is not None:
        # 加载 best model
        load_best_model_if_available(model, output_dir, device)

        print(f"\n{'='*60}")
        print(f"📊 独立测试集评估 (test.txt)")
        print(f"{'='*60}")
        test_results = validate(
            model,
            test_data,
            config,
            q_texts,
            tokenizer,
            q_to_kc_mapping,
            edge_index,
            use_amp,
        )
        print(f"测试结果:")
        for k, v in test_results.items():
            print(f"  {k}: {v:.4f}")

    save_training_summary(summary_path, best_epoch, best, test_results, split_info)


if __name__ == "__main__":
    main()
