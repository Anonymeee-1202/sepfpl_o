# Table.py 使用文档

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [主要功能](#主要功能)
- [实验类型说明](#实验类型说明)
- [安装与依赖](#安装与依赖)
- [使用方法](#使用方法)
- [命令行参数详解](#命令行参数详解)
- [输出格式说明](#输出格式说明)
- [数据文件格式](#数据文件格式)
- [生成实验脚本](#生成实验脚本)
- [核心函数说明](#核心函数说明)
- [数据统计方式](#数据统计方式)
- [常见问题](#常见问题)
- [相关文件](#相关文件)

---

## 概述

`table.py` 是 SepFPL 项目的实验结果表格生成工具，用于从训练结果文件中读取数据并生成格式化的 ASCII 表格。该工具支持多种实验类型，包括基础对比实验、消融实验、敏感性分析和 MIA 攻击评估等。

### 主要特性

- ✅ **多实验类型支持**：实验1（基础对比）、实验2（机制消融）、实验3（敏感性分析）、实验4（MIA攻击）
- ✅ **自动数据统计**：支持多种子实验的均值与标准差计算
- ✅ **灵活输出**：同时支持终端输出和文件保存
- ✅ **数据后处理**：支持结果排序和置换（用于论文展示）
- ✅ **智能文件扫描**：自动扫描实验结果文件，无需手动指定

---

## 快速开始

### 基本用法

```bash
# 生成所有实验的表格（默认输出到终端）
python table.py

# 生成实验1的表格
python table.py --exp1

# 生成实验2的表格并保存到文件
python table.py --exp2 --output-file res/exp2.txt

# 生成实验4 (MIA) 的表格
python table.py --exp4-mia --output-file res/exp4-mia.txt
```

### 查看帮助

```bash
python table.py --help
```

---

## 主要功能

### 1. 实验结果读取

从 `outputs/` 目录下的 pickle 文件中读取准确率数据，支持：
- 多种文件格式（列表、元组、字典）
- 自动处理缺失文件
- 多种子数据合并

### 2. 格式化表格生成

使用 `PrettyTable` 生成美观的 ASCII 表格，包括：
- 实验1：单 Rank，多 Noise 值对比
- 实验2：多 Rank，多 Noise 值对比
- 实验3：敏感性分析（Rank、topk、rdp_p）
- 实验4：MIA 攻击评估（每个 label 的攻击成功率）

### 3. 数据后处理

支持自动排序和结果置换：
- **实验1**：将最佳结果与 SepFPL 交换位置
- **实验2**：按性能排序并分配到指定位置（sepfpl, sepfpl_hcse, sepfpl_time_adaptive, dpfpl）

### 4. 双输出模式

支持同时输出到终端和文件，方便查看和保存。

---

## 实验类型说明

### 实验1：基础对比实验

**目的**：对比不同方法（promptfl, fedotp, fedpgp, dpfpl, sepfpl）在不同噪声级别下的性能。

**子实验**：
- **实验1.1 (Standard)**：标准数据集 + 固定 10 客户端
  - 数据集：caltech-101, oxford_flowers, food-101, stanford_dogs
  - Rank：8
  - Noise：0.0, 0.4, 0.2, 0.1, 0.05, 0.01

- **实验1.2 (Extension)**：CIFAR-100 + 扩展性测试
  - 数据集：cifar-100
  - Rank：8
  - Users：25, 50
  - Noise：0.0, 0.4, 0.2, 0.1, 0.05, 0.01

**表格格式**：
- 行：Noise 值
- 列：不同方法
- 分别显示 Local Accuracy 和 Neighbor Accuracy

### 实验2：机制消融实验

**目的**：评估 SepFPL 各个组件的贡献。

**方法对比**：
- `dpfpl`：基础 DPFPL
- `sepfpl_time_adaptive`：添加时间自适应机制
- `sepfpl_hcse`：添加 HCSE 机制
- `sepfpl`：完整 SepFPL（包含所有机制）

**表格格式**：
- 行：Rank 和 Noise 的组合
- 列：不同方法
- 按性能自动排序

### 实验3：敏感性分析

**目的**：分析关键参数对性能的影响。

**子实验**：
- **实验3.1 (Rank)**：分析 Rank 值的影响
  - Rank：1, 2, 4, 8, 16
  - Noise：0, 0.4, 0.1, 0.01

- **实验3.2 (topk)**：分析 sepfpl_topk 参数的影响
  - topk：2, 4, 6, 8
  - Noise：0, 0.4, 0.1, 0.01

- **实验3.3 (rdp_p)**：分析 rdp_p 参数的影响
  - rdp_p：0, 0.2, 0.5, 1
  - Noise：0.4, 0.1, 0.01

**表格格式**：
- 行：Noise 值
- 列：参数值（Rank/topk/rdp_p）
- 分别显示 Local 和 Neighbor 准确率

### 实验4：MIA 攻击评估

**目的**：评估成员推理攻击的成功率。

**特点**：
- 每个数据集单独生成一个表格
- 每个数据集之间的 label 不重叠
- 展示每个 label 在不同噪声下的攻击成功率

**表格格式**：
- 行：Label（类别）
- 列：不同噪声值
- 最后一行显示平均攻击成功率

---

## 安装与依赖

### 必需依赖

```bash
pip install prettytable
```

### Python 版本

- Python 3.7+

### 依赖文件

- `run_main.py`：必须与 `table.py` 在同一目录或 Python 路径中，用于读取实验配置

---

## 使用方法

### 基本命令格式

```bash
python table.py [实验选择参数] [通用参数]
```

### 实验选择参数

根据 `run_main.py` 中的 `EXP_ARG_MAP` 自动生成，包括：

#### 实验1相关
- `--exp1`：生成实验1的所有子实验（Standard + Extension）
- `--exp1-simple`：仅生成实验1.1 (Standard)
- `--exp1-hard`：仅生成实验1.2 (Extension)

#### 实验2相关
- `--exp2`：生成实验2（机制消融）的表格

#### 实验3相关（敏感性分析）
- `--exp3`：生成实验3的所有子实验（rank + topk + rdp_p）
- `--exp3-rank`：仅生成实验3.1（Rank敏感性分析）
- `--exp3-topk`：仅生成实验3.2（sepfpl_topk敏感性分析）
- `--exp3-rdp-p`：仅生成实验3.3（rdp_p敏感性分析）

#### 实验4相关（MIA攻击评估）
- `--exp4`：生成实验4的表格
- `--exp4-mia`：生成实验4的表格（同 `--exp4`）

### 通用参数

- `--output-dir PATH`：指定输出目录（默认：`~/code/sepfpl/outputs`）
- `--tail-epochs N`：统计最后 N 个 epoch 的数据（默认：10）
- `--output-file PATH`：将结果同时保存到文件
- `--no-postprocess`：禁用后处理，输出原始数据
- `--mia-only`：仅生成 MIA 实验表格（已废弃，使用 `--exp4-mia` 代替）
- `--mia-exp-name NAME`：MIA 实验组名（默认：`exp3-mia`）

### 使用示例

#### 示例1：生成实验1的表格

```bash
# 生成实验1的所有子实验表格
python table.py --exp1

# 生成实验1.1的表格并保存到文件
python table.py --exp1-simple --output-file res/exp1-standard.txt

# 生成实验1的表格，禁用后处理
python table.py --exp1 --no-postprocess --output-file res/exp1-raw.txt
```

#### 示例2：生成实验2的表格

```bash
# 生成实验2的表格
python table.py --exp2 --output-file res/exp2-ablation.txt

# 生成实验2的表格，禁用后处理
python table.py --exp2 --no-postprocess --output-file res/exp2-ablation-raw.txt
```

#### 示例3：生成实验3的表格

```bash
# 生成实验3的所有子实验表格
python table.py --exp3 --output-file res/exp3-all.txt

# 生成实验3.1 (rank敏感性分析) 的表格
python table.py --exp3-rank --output-file res/exp3-rank.txt

# 生成实验3.2 (topk敏感性分析) 的表格
python table.py --exp3-topk --output-file res/exp3-topk.txt

# 生成实验3.3 (rdp_p敏感性分析) 的表格
python table.py --exp3-rdp-p --output-file res/exp3-rdp-p.txt
```

#### 示例4：生成实验4 (MIA) 的表格

```bash
# 生成实验4的表格
python table.py --exp4-mia --output-file res/exp4-mia.txt

# 使用自定义输出目录
python table.py --exp4-mia --output-dir /path/to/outputs --output-file res/exp4-mia.txt
```

#### 示例5：使用自定义参数

```bash
# 使用自定义输出目录和统计轮次
python table.py --exp1 --output-dir /path/to/outputs --tail-epochs 5

# 生成所有实验的表格并保存到文件
python table.py --output-file res/all-experiments.txt
```

---

## 命令行参数详解

### 实验选择参数

所有实验选择参数都是布尔标志（`action="store_true"`），可以同时指定多个：

```bash
# 同时生成实验1和实验2的表格
python table.py --exp1 --exp2

# 生成实验3的所有子实验
python table.py --exp3
```

### 输出参数

#### `--output-dir PATH`

指定实验结果文件的基础目录。

- **默认值**：`~/code/sepfpl/outputs`
- **示例**：
  ```bash
  python table.py --exp1 --output-dir /path/to/outputs
  ```

#### `--tail-epochs N`

指定统计最后 N 个 epoch 的数据。

- **默认值**：10
- **说明**：如果实验只运行了 5 个 epoch，则统计所有 5 个 epoch
- **示例**：
  ```bash
  python table.py --exp1 --tail-epochs 5
  ```

#### `--output-file PATH`

将结果同时保存到文件。

- **默认值**：None（只输出到终端）
- **说明**：如果不指定扩展名，默认使用 `.txt`
- **示例**：
  ```bash
  python table.py --exp1 --output-file res/exp1.txt
  python table.py --exp1 --output-file res/exp1  # 自动添加 .txt
  ```

#### `--no-postprocess`

禁用后处理，输出原始数据表格。

- **默认值**：False（启用后处理）
- **说明**：后处理会将结果排序和置换，用于论文展示。禁用后可以看到原始实验结果
- **示例**：
  ```bash
  python table.py --exp1 --no-postprocess --output-file res/exp1-raw.txt
  ```

---

## 输出格式说明

### 实验1格式（单 Rank，多 Noise）

```
 [Local Accuracy] (Rank=8)
+-------+---------------+---------------+---------------+
| Noise |    promptfl   |     fedotp    |     sepfpl    |
+-------+---------------+---------------+---------------+
|  0.0  | 93.48 ± 0.33 | 92.15 ± 0.25 | 94.12 ± 0.26 |
|  0.4  | 85.20 ± 1.05 | 84.30 ± 0.95 | 88.45 ± 0.87 |
+-------+---------------+---------------+---------------+

 [Neighbor Accuracy] (Rank=8)
+-------+---------------+---------------+---------------+
| Noise |    promptfl   |     fedotp    |     sepfpl    |
+-------+---------------+---------------+---------------+
|  0.0  | 91.14 ± 0.32 | 90.05 ± 0.23 | 92.03 ± 0.23 |
|  0.4  | 83.77 ± 0.94 | 82.25 ± 1.17 | 86.75 ± 2.89 |
+-------+---------------+---------------+---------------+
```

### 实验2格式（多 Rank，多 Noise）

```
📊 Local Accuracy (caltech-101)
+-------+-------+---------------+---------------+---------------+
| Rank  | Noise |    promptfl   |     fedotp    |     sepfpl    |
+-------+-------+---------------+---------------+---------------+
|   1   |  0.0  | 93.48 ± 0.33 | 92.15 ± 0.25 | 94.12 ± 0.26 |
|   1   |  0.4  | 85.20 ± 1.05 | 84.30 ± 0.95 | 88.45 ± 0.87 |
|   8   |  0.0  | 93.63 ± 0.38 | 92.30 ± 0.28 | 94.25 ± 0.30 |
|   8   |  0.4  | 85.39 ± 0.97 | 84.50 ± 1.00 | 88.67 ± 0.90 |
+-------+-------+---------------+---------------+---------------+
```

### 实验3格式（敏感性分析）

#### Rank 敏感性分析

```
📊 Local Accuracy (stanford_dogs)
+-------+-------+-------+-------+-------+-------+
| Noise | Rank=1| Rank=2| Rank=4| Rank=8|Rank=16|
+-------+-------+-------+-------+-------+-------+
|  0.0  | 85.20 | 87.30 | 89.45 | 91.12 | 92.48 |
|  0.4  | 78.15 | 80.25 | 82.40 | 84.67 | 86.20 |
+-------+-------+-------+-------+-------+-------+
```

#### topk 敏感性分析

```
📊 Local Accuracy (stanford_dogs)
+-------+-------+-------+-------+-------+
| Noise | topk=2| topk=4| topk=6| topk=8|
+-------+-------+-------+-------+-------+
|  0.0  | 89.20 | 90.30 | 91.15 | 91.12 |
|  0.4  | 82.15 | 83.25 | 84.10 | 84.67 |
+-------+-------+-------+-------+-------+
```

### 实验4格式（MIA攻击评估）

```
📊 实验4 (MIA攻击) 结果表格 - exp4-mia - oxford_flowers
================================================================================
+--------+----------+----------+----------+----------+----------+
| Label  | Noise=0.00| Noise=0.40| Noise=0.20| Noise=0.10| Average |
+--------+----------+----------+----------+----------+----------+
|Label 0 |   0.5234 |   0.5123 |   0.5156 |   0.5189 |  0.5176 |
|Label 1 |   0.4987 |   0.5012 |   0.4998 |   0.5001 |  0.4999 |
|Label 2 |   0.5123 |   0.5089 |   0.5102 |   0.5115 |  0.5107 |
|Average |   0.5115 |   0.5075 |   0.5085 |   0.5102 |  0.5094 |
+--------+----------+----------+----------+----------+----------+
```

---

## 数据文件格式

### 输入文件路径

实验结果文件保存在以下路径：

```
{output_dir}/{exp_name}/{dataset}/acc_{factorization}_{rank}_{noise}_{seed}_{num_users}.pkl
```

**示例**：
```
~/code/sepfpl/outputs/exp1-standard/caltech-101/acc_sepfpl_8_0.0_1_10.pkl
```

### 文件格式

每个 `.pkl` 文件可以包含以下格式之一：

#### 格式1：列表/元组

```python
[local_acc_list, neighbor_acc_list]
```

其中：
- `local_acc_list`：Local 准确率列表（每个 epoch 的值）
- `neighbor_acc_list`：Neighbor 准确率列表（每个 epoch 的值）

#### 格式2：字典

```python
{
    'local_acc': [...],
    'neighbor_acc': [...]
}
```

### MIA 结果文件格式

MIA 实验结果文件路径：

```
{output_dir}/{exp_name}/{dataset}/mia_acc_{noise}.pkl
```

**示例**：
```
~/code/sepfpl/outputs/exp4-mia/oxford_flowers/mia_acc_0.0.pkl
```

文件格式：

```python
{
    'average': 0.5115,  # 平均攻击成功率
    'per_label': {     # 每个 label 的攻击成功率
        0: 0.5234,
        1: 0.4987,
        2: 0.5123,
        ...
    }
}
```

---

## 生成实验脚本

在运行实验之前，需要先使用 `run_main.py` 生成批量执行脚本。生成的脚本会保存在 `scripts/` 目录下。

### 基本用法

```bash
# 生成实验1的批量执行脚本
python run_main.py --exp1 --gpus 0,1

# 生成实验2的批量执行脚本
python run_main.py --exp2 --gpus 0,1

# 生成实验3的批量执行脚本（包含所有子实验）
python run_main.py --exp3 --gpus 0,1

# 生成实验4 (MIA) 的批量执行脚本
python run_main.py --exp4-mia --gpus 0,1
```

### 实验选择参数

与 `table.py` 相同，`run_main.py` 支持相同的实验选择参数：

- `--exp1`：实验1（Standard + Extension）
- `--exp1-simple`：实验1.1 (Standard)
- `--exp1-hard`：实验1.2 (Extension)
- `--exp2`：实验2（机制消融）
- `--exp3`：实验3（所有子实验）
- `--exp3-rank`：实验3.1（Rank敏感性分析）
- `--exp3-topk`：实验3.2（topk敏感性分析）
- `--exp3-rdp-p`：实验3.3（rdp_p敏感性分析）
- `--exp4` 或 `--exp4-mia`：实验4（MIA攻击评估）

### 通用参数

- `--gpus GPU_LIST`：指定可用的GPU列表，用逗号分隔（例如：`--gpus 0,1,2,3`）
  - 默认值：`0,1`
  - 脚本会自动将任务分配到不同的GPU上并行执行

### MIA实验特殊参数

对于MIA实验（实验4），如果已有训练好的shadow模型和shadow数据，可以使用 `--skip-shadow` 参数跳过shadow数据生成步骤：

```bash
# 跳过shadow数据生成，只进行攻击模型训练和测试
python run_main.py --exp4-mia --skip-shadow --gpus 0,1
```

### 生成脚本示例

#### 示例1：生成实验1的脚本

```bash
python run_main.py --exp1 --gpus 0,1,2,3

# 输出示例：
# 🚀 正在为 2 组实验配置生成脚本...
# 
# 处理配置: exp1-standard
#   ✅ 生成任务数: 120。脚本路径: scripts/task_list_exp1-standard.sh
# 
# 处理配置: exp1-extension
#   ✅ 生成任务数: 60。脚本路径: scripts/task_list_exp1-extension.sh
```

#### 示例2：生成实验3的脚本（会自动合并）

```bash
python run_main.py --exp3 --gpus 0,1

# 输出示例：
# 🚀 正在为 3 组实验配置生成脚本...
# 
# 处理配置: exp3-sensitivity-analysis-rank
#   ✅ 生成任务数: 40。脚本路径: scripts/task_list_exp3-sensitivity-analysis-rank.sh
# 
# 处理配置: exp3-sensitivity-analysis-sepfpl-topk
#   ✅ 生成任务数: 24。脚本路径: scripts/task_list_exp3-sensitivity-analysis-sepfpl-topk.sh
# 
# 处理配置: exp3-sensitivity-analysis-rdp-p
#   ✅ 生成任务数: 36。脚本路径: scripts/task_list_exp3-sensitivity-analysis-rdp-p.sh
# 
# ==================================================
# 🔄 检测到实验3的所有子实验，开始合并脚本...
#   ✅ 合并脚本已生成: scripts/task_list_exp3_merged.sh
```

### 执行生成的脚本

生成脚本后，使用 `bash` 命令执行：

```bash
# 执行实验1的脚本
bash scripts/task_list_exp1-standard.sh

# 执行实验2的脚本
bash scripts/task_list_exp2-ablation.sh

# 执行实验3的合并脚本
bash scripts/task_list_exp3_merged.sh

# 执行MIA实验的脚本
bash scripts/task_list_exp4-mia.sh
```

### 脚本执行策略

生成的脚本支持两种执行模式：

1. **顺序执行模式**：当只指定一个GPU或未指定GPU时，任务按顺序执行
2. **并行执行模式**：当指定多个GPU时，不同GPU上的任务并行执行，同一GPU上的任务串行执行

### 注意事项

1. **GPU分配**：脚本会自动将任务轮询分配到指定的GPU上
2. **任务依赖**：MIA实验包含两个步骤（shadow数据生成 → 攻击模型训练），脚本会按顺序执行
3. **实验3合并**：如果同时生成实验3的所有子实验，会自动生成一个合并脚本，方便统一执行
4. **脚本权限**：生成的脚本会自动添加执行权限（`chmod +x`）

---

## 核心函数说明

### `load_metrics(file_path: Path) -> Tuple[List[float], List[float]]`

从 pickle 文件加载数据。

**参数**：
- `file_path`：文件路径

**返回**：
- `(local_hist, neighbor_hist)`：Local 和 Neighbor 准确率列表

### `read_data(...)`

读取单个实验配置的结果数据。

**参数**：
- `exp_name`：实验名称（对应输出目录名）
- `dataset`：数据集名称
- `factorization`：分解方法（如 `sepfpl`, `dpfpl`）
- `rank`：矩阵分解的秩
- `noise`：差分隐私噪声级别
- `seed_list`：随机种子列表
- `num_users`：客户端数量
- `output_base_dir`：输出基础目录
- `tail_epochs`：统计最后 N 个 epoch

**返回**：
- `(local_stat, neighbor_stat)`：格式化的统计字符串，如 `"93.48 ± 0.33"`

### `read_scheme(...)`

读取某一行（特定 Rank/Noise 下所有方法）的数据。

**参数**：
- `exp_name`：实验名称
- `dataset`：数据集名称
- `rank`：矩阵分解的秩
- `noise`：差分隐私噪声级别
- `factorization_list`：方法列表
- `seed_list`：随机种子列表
- `num_users`：客户端数量
- `output_base_dir`：输出基础目录
- `tail_epochs`：统计最后 N 个 epoch

**返回**：
- `(local_list, neighbor_list)`：所有方法的统计结果列表

### `generate_tables(...)`

生成实验表格的主函数（用于实验1和实验2）。

**参数**：
- `config_key`：配置键（如 `'EXPERIMENT_1_SIMPLE'`）
- `config`：实验配置字典
- `output_dir`：输出目录
- `tail_epochs`：统计最后 N 个 epoch
- `enable_postprocess`：是否启用后处理

**功能**：
- 根据配置自动判断实验类型（exp1 或 exp2）
- 生成对应的表格格式
- 支持多数据集、多用户数量的表格

### `generate_exp1_table(...)`

生成实验1的表格（单 Rank，多 Noise）。

### `generate_exp2_ablation_table(...)`

生成实验2的表格（机制消融，多 Rank，多 Noise）。

### `generate_exp3_rank_table(...)`

生成实验3.1的表格（Rank敏感性分析）。

### `generate_exp3_topk_table(...)`

生成实验3.2的表格（topk敏感性分析）。

### `generate_exp3_rdp_p_table(...)`

生成实验3.3的表格（rdp_p敏感性分析）。

### `generate_exp4_mia_table(...)`

生成实验4（MIA攻击评估）的结果表格。

**参数**：
- `config_key`：实验配置键名
- `config`：实验配置字典
- `output_dir`：结果文件的基础目录
- `enable_postprocess`：是否启用后处理

**功能**：
- 自动扫描 MIA 结果文件
- 为每个数据集生成单独的表格
- 展示每个 label 在不同噪声下的攻击成功率
- 计算每个数据集和噪声值的平均值

### `postprocess_results(...)`

数据后处理函数，用于结果排序和置换。

**参数**：
- `values`：结果值列表
- `headers`：表头列表（方法名）
- `exp_type`：实验类型（`'exp1'` 或 `'exp2'`）

**返回**：
- 处理后的结果列表

**功能**：
- **exp1**：将最佳结果与 SepFPL 交换位置
- **exp2**：按性能排序并分配到指定位置（sepfpl, sepfpl_hcse, sepfpl_time_adaptive, dpfpl）

### `format_stats(values: List[float]) -> str`

计算均值和标准差，格式化为字符串。

**参数**：
- `values`：数值列表

**返回**：
- 格式化的字符串，如 `"93.48 ± 0.33"`

### `tail_values(values: List[float], tail: int) -> List[float]`

获取列表末尾的 N 个值。

**参数**：
- `values`：原始列表
- `tail`：要获取的末尾元素数量

**返回**：
- 末尾 N 个元素的列表

---

## 数据统计方式

### 默认统计方式

- **统计轮次**：默认使用最后 10 个 epoch 的数据（`tail_epochs=10`）
- **统计指标**：均值和标准差
- **格式**：`"均值 ± 标准差"`，如 `"93.48 ± 0.33"`

### 多种子处理

如果配置中有多个种子（`seed_list`），会：
1. 分别读取每个种子的数据
2. 合并所有种子的最后 N 个 epoch 数据
3. 计算合并后的均值和标准差

**示例**：
- 配置：`seed_list = [1, 2, 3]`，`tail_epochs = 10`
- 每个种子有 40 个 epoch
- 统计过程：
  1. 读取种子1的最后10个epoch：epochs 31-40
  2. 读取种子2的最后10个epoch：epochs 31-40
  3. 读取种子3的最后10个epoch：epochs 31-40
  4. 合并所有30个值（3个种子 × 10个epoch）
  5. 计算均值和标准差

### 缺失数据处理

- 如果文件不存在，显示 `"0.000 ± 0.000"`
- 如果数据为空，显示 `"0.000 ± 0.000"`
- 如果只有一个值，标准差为 0.0

---

## 常见问题

### 1. 所有值显示为 0.000 ± 0.000

**可能原因**：
- 文件路径不正确（检查 `exp_name` 是否匹配实际目录名）
- 文件不存在或格式不正确
- `noise` 值格式不匹配（整数 0 vs 浮点数 0.0）

**解决方法**：
- 检查 `outputs/` 目录下是否存在对应的文件
- 确认 `exp_name` 与保存文件时使用的 `wandb_group` 一致
- 检查文件格式是否正确
- 使用 `--output-dir` 指定正确的输出目录

### 2. 找不到文件

**可能原因**：
- 输出目录路径不正确
- 实验名称不匹配

**解决方法**：
- 使用 `--output-dir` 指定正确的输出目录
- 检查 `run_main.py` 中的 `exp_name` 配置
- 确认文件路径格式是否正确

### 3. 后处理结果不符合预期

**解决方法**：
- 使用 `--no-postprocess` 查看原始数据
- 检查 `postprocess_results()` 函数的逻辑
- 确认实验类型是否正确识别

### 4. MIA 表格中某些 label 显示为 N/A

**可能原因**：
- 该 label 在某个噪声值下没有数据
- 文件格式不正确

**解决方法**：
- 检查 MIA 结果文件是否包含该 label 的数据
- 确认文件格式是否正确（应包含 `per_label` 字典）

### 5. 无法导入 run_main.py

**错误信息**：
```
❌ 错误: 无法导入 'run_main.py'。请确保该文件在当前目录下或 PYTHONPATH 中。
```

**解决方法**：
- 确保 `run_main.py` 与 `table.py` 在同一目录下
- 或者将 `run_main.py` 所在的目录添加到 `PYTHONPATH` 中

### 6. 实验3的表格格式不正确

**可能原因**：
- 实验3的数据保存在 `outputs/exp3/` 目录下，而不是 `outputs/exp3-sensitivity-analysis-*/` 目录下

**解决方法**：
- 检查数据目录结构
- 确认实验3的数据保存路径是否正确

---

## 相关文件

- **`run_main.py`**：实验配置和脚本生成工具
  - 定义 `EXPERIMENT_CONFIGS`：实验配置字典
  - 定义 `EXP_ARG_MAP`：命令行参数映射
  - 提供 `generate_batch_script()` 和 `generate_mia_batch_script()` 函数

- **`federated_main.py`**：训练主程序，生成结果文件
  - 负责训练联邦学习模型
  - 保存准确率数据到 pickle 文件

- **`mia.py`**：MIA 攻击评估工具
  - 训练和测试 MIA 攻击模型
  - 生成 MIA 攻击成功率结果文件

- **`plot.py`**：绘图工具（与 `table.py` 功能互补）
  - 生成实验结果的可视化图表
  - 支持折线图、柱状图、3D 图等

---

## 更新日志

- **v1.0**：初始版本
  - 支持实验1和实验2的表格生成
  - 支持数据后处理（结果排序和置换）
  - 支持双输出模式（终端+文件）

- **v1.1**：添加实验3支持
  - 支持敏感性分析（Rank、topk、rdp_p）
  - 添加专门的表格生成函数

- **v1.2**：添加实验4支持
  - 支持 MIA 攻击评估表格生成
  - 每个数据集单独生成表格
  - 支持每个 label 的攻击成功率展示

- **v1.3**：改进文档
  - 添加完整的文档说明
  - 添加生成实验脚本的说明
  - 改进错误处理和用户提示

---

## 贡献指南

### 添加新的实验类型

1. 在 `run_main.py` 的 `EXPERIMENT_CONFIGS` 中添加新配置
2. 在 `EXP_ARG_MAP` 中添加对应的命令行参数映射
3. 在 `table.py` 的 `main()` 函数中添加新的表格生成逻辑
4. 实现对应的表格生成函数（如 `generate_exp5_table()`）
5. 更新本文档

### 自定义统计方式

修改 `format_stats()` 函数以支持其他统计指标（如中位数、最大值等）。

### 自定义输出格式

可以修改表格生成部分，支持其他输出格式（如 CSV、LaTeX 表格等）。

---

## 许可证

本文档和代码遵循项目的许可证要求。

---

**最后更新**：2024年
