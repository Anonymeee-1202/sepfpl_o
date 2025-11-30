# MIA (Membership Inference Attack) 完整执行流程指南

本文档详细说明如何执行完整的 MIA 攻击流程，包括数据准备、模型训练和测试。

## 📋 目录

1. [流程概述](#流程概述)
2. [快速开始：使用批量脚本](#快速开始使用批量脚本)
3. [步骤 1: 生成 Shadow 数据](#步骤-1-生成-shadow-数据)
4. [步骤 2: 训练 MIA 攻击模型](#步骤-2-训练-mia-攻击模型)
5. [步骤 3: 测试 MIA 攻击模型](#步骤-3-测试-mia-攻击模型)
6. [完整示例](#完整示例)
7. [文件路径说明](#文件路径说明)

---

## 流程概述

完整的 MIA 攻击流程包括以下三个步骤：

```
1. 生成 Shadow 数据 (srun_generate_shadow.sh)
   - 自动训练多个 shadow 模型（seed 从 start_seed 到 end_seed）
   - 收集每个 shadow 模型的预测结果
   ↓
2. 训练 MIA 攻击模型 (mia_train.py)
   - 使用 shadow 数据训练攻击模型
   ↓
3. 测试 MIA 攻击模型 (mia_test.py)
   - 对目标模型进行 MIA 攻击测试
```

**注意**: `srun_generate_shadow.sh` 脚本已经包含了训练 shadow 模型的过程，因此不需要单独训练目标模型。如果目标模型的 seed 在 shadow 模型的 seed 范围内，可以直接使用对应的 shadow 模型作为目标模型。

---

## 快速开始：使用批量脚本

推荐使用 `run_main.py` 生成批量执行脚本，自动处理所有步骤：

```bash
# 生成 exp3-mia 实验的批量脚本
python run_main.py --exp3-mia --gpus 0,1

# 执行生成的脚本
bash scripts/task_list_exp3-mia.sh
```

生成的脚本会自动执行以下三个步骤：
1. 生成 Shadow 数据（包含训练 shadow 模型）
2. 训练 MIA 攻击模型
3. 测试 MIA 攻击模型

---

## 步骤 1: 生成 Shadow 数据

Shadow 数据用于训练 MIA 攻击模型。`srun_generate_shadow.sh` 脚本会自动完成以下工作：
1. 训练多个 shadow 模型（seed 从 `start-seed` 到 `end-seed`）
2. 为每个 shadow 模型生成预测数据并保存

**注意**: 该脚本会自动跳过测试阶段（使用 `--skip-test`）以加快训练速度。

### 使用脚本执行

```bash
bash srun_generate_shadow.sh \
  <root> \
  <dataset-config-file> \
  <num-users> \
  <factorization> \
  <rank> \
  <noise> \
  <start-seed> \
  <end-seed> \
  <round> \
  [wandb-group]
```

### 示例

```bash
# 生成 seed 0-9 的 shadow 数据（共 10 个 shadow 模型）
bash srun_generate_shadow.sh \
  /home/liuxin25/dataset \
  configs/datasets/caltech101.yaml \
  10 \
  sepfpl \
  8 \
  0.1 \
  0 \
  9 \
  20 \
  exp3-mia
```

### 工作原理

对于每个 seed（从 `start-seed` 到 `end-seed`），脚本会：
1. 训练 shadow 模型（使用 `federated_main.py`，自动添加 `--skip-test`）
2. 生成 shadow 数据（使用 `federated_main.py --generate-shadow`）

所有 shadow 模型都会自动跳过测试阶段，以加快训练速度。

### 输出文件

- **Shadow 数据**: `~/data/sepfpl/outputs/{wandb_group}/{dataset}/shadow_{noise}_{seed}.pkl`
  - 每个文件包含一个列表，每个元素是 `(prediction, membership, label)` 元组
  - `prediction`: 模型对样本的预测概率分布
  - `membership`: 1 表示训练集样本，0 表示测试集样本
  - `label`: 样本的真实标签

---

## 步骤 2: 训练 MIA 攻击模型

使用生成的 shadow 数据训练 MIA 攻击模型。攻击模型会为每个类别训练一个二分类器。

### 使用脚本执行

```bash
bash srun_mia.sh train \
  <root> \
  <dataset-config-file> \
  <noise> \
  <seed> \
  [wandb-group]
```

### 示例

```bash
# 训练 MIA 攻击模型
bash srun_mia.sh train \
  /home/liuxin25/dataset \
  configs/datasets/caltech101.yaml \
  0.1 \
  1 \
  default
```

### 输出文件

- **攻击模型**: `~/data/sepfpl/checkpoints/{wandb_group}/{dataset}/mia_{label}_{noise}.pth.tar`
  - 每个类别一个模型文件

---

## 步骤 3: 测试 MIA 攻击模型

使用训练好的攻击模型对目标模型进行 MIA 攻击测试。

### 使用脚本执行

```bash
bash srun_mia.sh test \
  <root> \
  <dataset-config-file> \
  <num-users> \
  <factorization> \
  <rank> \
  <noise> \
  <seed> \
  <round> \
  [wandb-group]
```

### 示例

```bash
# 测试 MIA 攻击模型
bash srun_mia.sh test \
  /home/liuxin25/dataset \
  configs/datasets/caltech101.yaml \
  10 \
  dpfpl \
  8 \
  0.1 \
  1 \
  40 \
  default
```

### 输出文件

- **攻击准确率**: `~/data/sepfpl/outputs/{wandb_group}/{dataset}/mia_acc_{noise}.pkl`
  - 包含平均 MIA 攻击成功率

---

## 完整示例

以下是一个完整的执行示例，展示如何对 caltech101 数据集进行 MIA 攻击评估。

### 方法 1: 使用批量脚本（推荐）

```bash
# 生成批量执行脚本
python run_main.py --exp3-mia --gpus 0,1

# 执行生成的脚本
bash scripts/task_list_exp3-mia.sh
```

### 方法 2: 手动执行

#### 1. 生成 Shadow 数据

```bash
# 生成 seed 0-9 的 shadow 数据（共 10 个 shadow 模型）
bash srun_generate_shadow.sh \
  /home/liuxin25/dataset \
  configs/datasets/caltech101.yaml \
  10 \
  sepfpl \
  8 \
  0.1 \
  0 \
  9 \
  20 \
  exp3-mia
```

**注意**: 如果目标模型的 seed（例如 seed=1）在 shadow 模型的 seed 范围内（0-9），则对应的 shadow 模型可以直接作为目标模型使用。

#### 2. 训练 MIA 攻击模型

```bash
bash srun_mia.sh train \
  /home/liuxin25/dataset \
  configs/datasets/caltech101.yaml \
  0.1 \
  1 \
  exp3-mia
```

#### 3. 测试 MIA 攻击模型

```bash
bash srun_mia.sh test \
  /home/liuxin25/dataset \
  configs/datasets/caltech101.yaml \
  10 \
  sepfpl \
  8 \
  0.1 \
  1 \
  20 \
  exp3-mia
```

---

## 文件路径说明

### 目录结构

```
~/data/sepfpl/
├── checkpoints/
│   └── {wandb_group}/
│       └── {dataset}/
│           ├── {factorization}_{rank}_{noise}_{seed}_{num_users}.pth.tar  # 目标模型
│           └── mia_{label}_{noise}.pth.tar  # 攻击模型（每个类别一个）
└── outputs/
    └── {wandb_group}/
        └── {dataset}/
            ├── shadow_{noise}_{seed}.pkl  # Shadow 数据（每个 seed 一个）
            ├── acc_{factorization}_{rank}_{noise}_{seed}_{num_users}.pkl  # 精度曲线
            └── mia_acc_{noise}.pkl  # MIA 攻击准确率
```

### 参数说明

- `{wandb_group}`: 实验分组名，默认为 "default"
- `{dataset}`: 数据集名称（从 dataset-config-file 提取）
- `{factorization}`: 分解方法（如 dpfpl, sepfpl）
- `{rank}`: 矩阵分解的秩
- `{noise}`: 差分隐私噪声级别
- `{seed}`: 随机种子
- `{num_users}`: 客户端数量
- `{label}`: 类别标签

---

## 注意事项

1. **Shadow 数据生成时间**: 生成多个 shadow 模型可能需要较长时间，建议使用 `run_main.py` 生成的批量脚本进行并行执行。

2. **参数一致性**: 确保所有步骤中使用的参数（dataset, factorization, rank, noise, seed, num_users, round）保持一致。

3. **wandb-group**: 如果使用不同的 wandb-group，需要确保所有步骤使用相同的 group 名称。

4. **GPU 资源**: Shadow 数据生成和 MIA 训练可以并行执行多个任务，建议合理分配 GPU 资源。

5. **目标模型**: 如果目标模型的 seed 在 shadow 模型的 seed 范围内，可以直接使用对应的 shadow 模型作为目标模型，无需单独训练。

6. **跳过测试**: `srun_generate_shadow.sh` 会自动为所有 shadow 模型添加 `--skip-test` 参数，以加快训练速度。

---

## 故障排查

### 问题 1: Shadow 数据文件未找到

**错误信息**: `Warning: shadow file not found: ...`

**解决方案**: 
- 确保已执行步骤 2 生成 shadow 数据
- 检查文件路径和 wandb-group 是否一致
- 确认 noise 和 seed 参数匹配

### 问题 2: 目标模型检查点未找到

**错误信息**: 在 `mia_test.py` 中无法加载目标模型

**解决方案**:
- 确保目标模型的 seed 在 shadow 模型的 seed 范围内（例如，如果 shadow seed 范围是 0-9，目标模型 seed 应该是 0-9 中的一个）
- 如果目标模型 seed 不在 shadow seed 范围内，需要单独训练目标模型
- 检查检查点文件路径
- 确认所有参数（factorization, rank, noise, seed, num_users）匹配

### 问题 3: 攻击模型未找到

**错误信息**: 在 `mia_test.py` 中无法加载攻击模型

**解决方案**:
- 确保已执行步骤 3 训练攻击模型
- 检查攻击模型文件路径
- 确认 noise 参数匹配

---

## 快速参考

### 脚本参数对照表

| 脚本 | 必需参数 | 可选参数 | 说明 |
|------|---------|---------|------|
| `srun_generate_shadow.sh` | root, dataset-config-file, num-users, factorization, rank, noise, start-seed, end-seed, round | wandb-group | 自动训练 shadow 模型并生成数据（跳过测试） |
| `srun_mia.sh train` | root, dataset-config-file, noise, seed | wandb-group | 训练 MIA 攻击模型 |
| `srun_mia.sh test` | root, dataset-config-file, num-users, factorization, rank, noise, seed, round | wandb-group | 测试 MIA 攻击模型 |

### 使用 run_main.py 生成批量脚本

```bash
# 生成 exp3-mia 实验的批量脚本
python run_main.py --exp3-mia --gpus 0,1

# 查看生成的脚本
cat scripts/task_list_exp3-mia.sh

# 执行生成的脚本
bash scripts/task_list_exp3-mia.sh
```

生成的脚本会自动处理所有步骤，包括 GPU 分配和错误处理。

---

## 联系与支持

如有问题，请检查：
1. 日志文件：`~/data/sepfpl/logs/`
2. 代码注释和文档
3. 项目 README

