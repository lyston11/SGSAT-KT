# TriSG-KT 项目规则

## 核心原则

### 1. 两段式训练（强制）
```bash
./scripts/1_precompute.sh    # 预计算（一次性）
./scripts/2_train.sh full    # 训练（重复）
```

### 2. 总运行脚本（推荐给新用户）
```bash
./scripts/train.sh full    # 自动预计算 + 训练
```

### 3. 每个Python脚本都要有对应的Shell脚本
- `1_precompute.py` → `1_precompute.sh`
- `2_train.py` → `2_train.sh`

### 4. 调参在配置文件
编辑 `configs/default.yaml` 调参。

### 5. 文档精简
只保留核心文档，删除冗余内容。

### 6. ⚠️ 删除检查规则（重要）
**删除任何文件前，必须检查是否有其他文件引用它**

```bash
# 检查引用
grep -r "filename" project/ --include="*.py" --include="*.md" --include="*.sh"

# 确认无引用后再删除
rm filename
```

### 7. Git 分支规则（强制）

**默认禁止直接在 `main` 上开发。**

标准流程：

```bash
git switch main
git pull
git switch -c feat/task-name
```

规则：
- 每个任务必须新建独立分支，不同类型改动不要混在一个分支里
- 分支命名统一使用：`feat/*`、`fix/*`、`docs/*`、`exp/*`、`release/*`
- 改完后先做最小验证，再提交，再合并回 `main`
- 分支合并完成后删除本地已完成分支；下一个任务重新创建新分支
- 远程分支默认保留，只有用户明确要求时才删除

### 8. 训练运行期间的 Git 规则（强制）

如果当前工作目录正在跑训练、预计算或长时间实验：

- **禁止** 在该工作目录执行 `git switch`、`git checkout`、`git merge`、`git rebase`
- 需要继续开发时，必须使用 `git worktree`

示例：

```bash
git worktree add ../TriSG-KT-dev -b feat/task-name main
```

### 9. 合并到 main 前的检查

合并前必须满足：
- 代码已通过语法检查或最小运行验证
- 训练/预处理改动已做对应 smoke test 或最小链路验证
- 相关文档已同步更新
- 已清理 `__pycache__`、`.pyc`、临时文件
- 每次执行结束后，必须主动清理本次创建的临时文件、缓存文件和中间产物，只保留用户明确需要的结果文件

推荐合并方式：

```bash
git switch main
git pull
git merge --no-ff feat/task-name
git push origin main
git branch -d feat/task-name
```

---

**每次开发前读一遍！**
