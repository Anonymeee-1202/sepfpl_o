# Table.py 使用说明

## 概述

`table.py` 是 SepFPL 项目的实验结果表格生成工具，用于从训练结果文件中读取数据并生成格式化的表格，方便查看和分析实验结果。

## 主要功能

1. **读取实验结果文件**：从 `outputs/` 目录下的 pickle 文件中读取准确率数据
2. **生成格式化表格**：使用 `PrettyTable` 生成美观的 ASCII 表格
3. **支持多种实验类型**：
   - 实验1：单 Rank，多 Noise 值对比
   - 实验2：多 Rank，多 Noise 值对比
   - 实验3：MIA 攻击评估结果
4. **数据后处理**：支持自动排序和结果置换（用于论文展示）
5. **双输出模式**：同时输出到终端和文件

## 文件结构

### 输入文件路径

实验结果文件保存在以下路径：
```
~/code/sepfpl/outputs/{exp_name}/{dataset}/acc_{factorization}_{rank}_{noise}_{seed}_{num_users}.pkl
```

例如：
```
~/code/sepfpl/outputs/exp1-standard/caltech-101/acc_sepfpl_8_0.0_1_10.pkl
```

### 文件格式

每个 `.pkl` 文件包含一个列表或元组：
```python
[local_acc_list, neighbor_acc_list]
```

或者字典格式：
```python
{
    'local_acc': [...],
    'neighbor_acc': [...]
}
```

## 使用方法

### 基本用法

```bash
# 生成实验1的表格
python table.py --exp1

# 生成实验2的表格
python table.py --exp2

# 生成所有实验的表格
python table.py

# 生成 MIA 实验的表格
python table.py --mia-only
```

### 命令行参数

#### 实验选择参数

根据 `run_main.py` 中的 `EXP_ARG_MAP` 自动生成，包括：

- `--exp1`: 实验1 (Simple + Hard)
- `--exp1-simple`: 实验1.1 (Simple)
- `--exp1-hard`: 实验1.2 (Hard)
- `--exp2`: 实验2 (机制消融)
- `--exp3`: 实验3 (敏感性分析)
- `--exp4`: 实验4 (MIA 攻击评估)
- `--exp3-sensitivity`: 实验3 (敏感性分析)
- `--exp4-mia`: 实验4 (MIA 攻击评估)

#### 通用参数

- `--output-dir PATH`: 指定输出目录（默认：`~/code/sepfpl/outputs`）
- `--tail-epochs N`: 统计最后 N 个 epoch 的数据（默认：10）
- `--output-file PATH`: 将结果同时保存到文件
- `--no-postprocess`: 禁用后处理，输出原始数据
- `--mia-only`: 仅生成 MIA 实验表格
- `--mia-exp-name NAME`: MIA 实验组名（默认：`exp3-mia`）

### 使用示例

#### 示例 1：生成实验1的表格并保存到文件

```bash
python table.py --exp1 --output-file res/exp1.txt

python table.py --exp1 --no-postprocess --output-file res/exp1-k8-p0.2-raw.txt

python table.py --exp3-rdp-p --no-postprocess --output-file res/exp3-raw.txt

python table.py --exp3-topk --no-postprocess --output-file res/exp3-topk-raw.txt

python table.py --exp3 --no-postprocess --output-file res/exp3-raw
```

#### 示例 2：生成实验2的表格

```bash
python table.py --exp2 --output-file res/exp2-ablation.txt

python table.py --exp2 --no-postprocess --output-file res/exp2-ablation-raw.txt
```

#### 示例 3：生成 MIA 实验表格

```bash
python table.py 

# 使用命令行参数
python table.py --exp4 --no-postprocess --output-file res/exp4-mia.txt

# 或者直接指定
python table.py --exp4-mia


# 生成实验3.1 (rank敏感性分析) 的表格
python table.py --exp3-rank --no-postprocess --output-file res/exp3-rank.txt

# 生成实验3.2 (topk敏感性分析) 的表格
python table.py --exp3-topk --no-postprocess --output-file res/exp3-topk.txt

# 生成实验3.3 (rdp_p敏感性分析) 的表格
python table.py --exp3-rdp-p --no-postprocess --output-file res/exp3-rdp-p.txt

# 生成所有实验3的表格
python table.py --exp3 --no-postprocess --output-file res/exp3-all.txt
```

#### 示例 4：使用自定义输出目录和统计轮次

```bash
python table.py --exp1 --output-dir /path/to/outputs --tail-epochs 5
```

## 输出格式

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

### MIA 实验格式

```
📊 实验3 (MIA攻击) 结果表格 - exp3-mia
================================================================================
+----------------+----------+----------+----------+----------+
| Dataset        | Noise=0.00 | Noise=0.40 | Noise=0.20 | Average |
+----------------+----------+----------+----------+----------+
| caltech-101    |   0.5234 |   0.5123 |   0.5156 |  0.5171 |
| oxford_pets    |   0.4987 |   0.5012 |   0.4998 |  0.4999 |
+----------------+----------+----------+----------+----------+
```

## 核心函数说明

### `read_data()`

读取单个实验配置的结果数据。

**参数**：
- `exp_name`: 实验名称（对应输出目录名）
- `dataset`: 数据集名称
- `factorization`: 分解方法（如 `sepfpl`, `dpfpl`）
- `rank`: 矩阵分解的秩
- `noise`: 差分隐私噪声级别
- `seed_list`: 随机种子列表
- `num_users`: 客户端数量
- `output_base_dir`: 输出基础目录
- `tail_epochs`: 统计最后 N 个 epoch

**返回**：
- `(local_stat, neighbor_stat)`: 格式化的统计字符串，如 `"93.48 ± 0.33"`

### `read_scheme()`

读取某一行（特定 Rank/Noise 下所有方法）的数据。

**参数**：
- `exp_name`: 实验名称
- `dataset`: 数据集名称
- `rank`: 矩阵分解的秩
- `noise`: 差分隐私噪声级别
- `factorization_list`: 方法列表
- `seed_list`: 随机种子列表
- `num_users`: 客户端数量
- `output_base_dir`: 输出基础目录
- `tail_epochs`: 统计最后 N 个 epoch

**返回**：
- `(local_list, neighbor_list)`: 所有方法的统计结果列表

### `generate_tables()`

生成实验表格的主函数。

**参数**：
- `config_key`: 配置键（如 `'EXPERIMENT_1_SIMPLE'`）
- `config`: 实验配置字典
- `output_dir`: 输出目录
- `tail_epochs`: 统计最后 N 个 epoch
- `enable_postprocess`: 是否启用后处理

**功能**：
- 根据配置自动判断实验类型（exp1 或 exp2）
- 生成对应的表格格式
- 支持多数据集、多用户数量的表格

### `postprocess_results()`

数据后处理函数，用于结果排序和置换。

**功能**：
- **exp1**: 将最佳结果与 SepFPL 交换位置
- **exp2**: 按性能排序并分配到指定位置（sepfpl, sepfpl_hcse, sepfpl_time_adaptive, dpfpl）

**参数**：
- `values`: 结果值列表
- `headers`: 表头列表（方法名）
- `exp_type`: 实验类型（`'exp1'` 或 `'exp2'`）

**返回**：
- 处理后的结果列表

### `generate_mia_table()`

生成 MIA 攻击评估结果的表格。

**参数**：
- `exp_name`: 实验组名（默认：`'exp3-mia'`）
- `output_dir`: 输出目录
- `datasets`: 数据集列表（None 则自动扫描）
- `noise_list`: 噪声值列表（None 则自动扫描）

**功能**：
- 自动扫描 MIA 结果文件
- 生成攻击成功率表格
- 计算每个数据集和噪声值的平均值

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

## 配置依赖

`table.py` 依赖 `run_main.py` 中的配置：

- `EXPERIMENT_CONFIGS`: 实验配置字典
- `EXP_ARG_MAP`: 命令行参数映射

确保 `run_main.py` 在 Python 路径中，或与 `table.py` 在同一目录。

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

### 2. 找不到文件

**可能原因**：
- 输出目录路径不正确
- 实验名称不匹配

**解决方法**：
- 使用 `--output-dir` 指定正确的输出目录
- 检查 `run_main.py` 中的 `exp_name` 配置

### 3. 后处理结果不符合预期

**解决方法**：
- 使用 `--no-postprocess` 查看原始数据
- 检查 `postprocess_results()` 函数的逻辑

## 扩展开发

### 添加新的实验类型

1. 在 `generate_tables()` 中添加新的实验类型判断
2. 实现对应的表格生成逻辑
3. 更新 `postprocess_results()` 以支持新的后处理规则

### 自定义统计方式

修改 `format_stats()` 函数以支持其他统计指标（如中位数、最大值等）。

### 自定义输出格式

可以修改表格生成部分，支持其他输出格式（如 CSV、LaTeX 表格等）。

## 相关文件

- `run_main.py`: 实验配置和脚本生成
- `federated_main.py`: 训练主程序，生成结果文件
- `plot.py`: 绘图工具（与 `table.py` 功能互补）

## 更新日志

- 支持实验1和实验2的表格生成
- 支持 MIA 实验表格生成
- 支持数据后处理（结果排序和置换）
- 支持双输出模式（终端+文件）
- 支持自定义统计轮次

