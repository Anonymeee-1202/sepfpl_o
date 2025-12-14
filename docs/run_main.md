以下是基于您提供的 `run_main.py` 使用指南整理成的 Markdown 说明文档。您可以直接将其保存为 `README.md` 或 `USAGE.md`。

-----

# SepFPL 实验运行指南

本文档详细说明了如何使用 `run_main.py` 脚本来生成和执行 SepFPL 项目的各类实验任务。

> **⚠️ 重要提示**
>
> `run_main.py` 的核心作用是**生成 Shell 执行脚本**（通常位于 `scripts/` 目录下），而非直接运行实验训练代码。
>
> **标准流程：**
>
> 1.  运行 `python run_main.py [参数]` 生成 `.sh` 脚本。
> 2.  运行生成的脚本（例如 `bash scripts/batch_run.sh`）开始实际训练。

-----

## 1\. 环境准备与数据下载

如果是首次运行本项目，请先下载实验所需的标准数据集。

```bash
python run_main.py --download
```

-----

## 2\. 实验 1：标准性能与扩展性 (Standard & Extension)

包含基础性能对比（`EXPERIMENT_1_STANDARD`）与大规模扩展性测试（`EXPERIMENT_1_EXTENSION`）。

### 常规生成

使用 GPU 0 和 1 生成任务脚本：

```bash
python run_main.py --exp1 --gpus "0,1"
```

*生成后请运行提示的脚本，通常为：* `bash scripts/batch_run.sh`

### 高并发模式

如果显存充足，可以使用多线程并行（例如每张 GPU 同时跑 2 个任务）以加快实验速度：

```bash
python run_main.py --exp1 --gpus "0,1,2,3" --threads 2
```

-----

## 3\. 实验 2：消融实验 (Ablation Study)

生成机制消融实验的任务（验证不同组件的有效性）。

```bash
python run_main.py --exp2 --gpus "0,1"
```

*生成后请运行：* `bash scripts/run_exp2-ablation.sh`

-----

## 4\. 实验 3：参数敏感性分析 (Sensitivity Analysis)

该模块包含 Rank、TopK 和 RDP\_P 参数的敏感性测试。脚本内部已包含自动合并与去重逻辑。

### 生成所有敏感性分析任务 (推荐)

将三个子实验合并到一个脚本中并去重：

```bash
python run_main.py --exp3 --gpus "0,1"
```

*生成后请运行：* `bash scripts/run_exp3_merged.sh`

### 仅运行特定子实验

例如，仅测试 Rank 的敏感性：

```bash
python run_main.py --exp3-rank
```

-----

## 5\. 实验 4：成员推断攻击 (MIA)

MIA 实验流程较长，支持**分阶段控制**以方便调试和断点续跑。

### 一次性生成完整流程 (默认)

默认包含生成 Shadow 数据 (`generate-shadow`) 和训练攻击模型 (`attack-train`)：

```bash
python run_main.py --exp4 --gpus "0,1"

python run_main.py --exp4 --gpus "0,1" --fed-train --generate-shadow --attack-train --attack-test --threads 2

python run_main.py --exp4 --gpus "0,1" --generate-shadow --attack-train --attack-test --threads 2

python run_main.py --exp4 --gpus "0,1" --attack-train --attack-test --threads 2

python run_main.py --exp4 --gpus "0,1" --attack-test --threads 2

```

### 分阶段生成 (调试推荐)

您可以根据需要单独生成某个阶段的任务：

  * **阶段 1：联邦训练目标模型 (Target Models)**

    ```bash
    python run_main.py --exp4 --fed-train
    ```

  * **阶段 2：生成 Shadow 数据**

    ```bash
    python run_main.py --exp4 --generate-shadow
    ```

  * **阶段 3：训练攻击模型 (Attack Models)**

    ```bash
    python run_main.py --exp4 --attack-train
    ```

  * **阶段 4：仅测试攻击模型**

    ```bash
    python run_main.py --exp4 --attack-test
    ```

-----

## 6\. 单任务测试模式 (Debug Mode)

如果您不想生成脚本文件，只想立即在当前终端运行一个具体的任务以测试代码是否报错，请使用 `--test` 模式。

```bash
python run_main.py --test \
    --dataset caltech-101 \
    --users 10 \
    --factorization sepfpl \
    --rank 8 \
    --noise 0.1 \
    --seed 1 \
    --gpus "0"
```

-----

## 7\. 常用组合示例

**场景：全量实验并行跑**
使用 4 张 GPU 卡，并行生成所有实验（Exp 1, 2, 3, 4），且每张卡开启 2 个线程并发，最大化利用资源。

**步骤 1：生成脚本**

```bash
python run_main.py \
    --exp1 --exp2 --exp3 --exp4 \
    --gpus "0,1,2,3" \
    --threads 2
```

**步骤 2：执行脚本**
终端会输出生成的脚本路径（通常是 `scripts/batch_run.sh`），执行它：

```bash
bash scripts/batch_run.sh
```