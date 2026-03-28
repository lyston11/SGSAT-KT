# SGSAT-KT 项目规则

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

---

**每次开发前读一遍！**
