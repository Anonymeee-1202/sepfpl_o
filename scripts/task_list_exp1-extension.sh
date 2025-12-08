#!/bin/bash

# 实验任务列表: exp1-extension
# 生成时间: 2025-12-05 17:19:26
# 任务总数: 12
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 顺序执行模式 (无 GPU 或单 GPU)
echo '▶️  正在执行任务 [1/12]: cifar-100 | sepfpl | r=8 | n=0.0 | u=25 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.0 1 40 exp1-extension '[1/12]' 8 0.2

echo '▶️  正在执行任务 [2/12]: cifar-100 | sepfpl | r=8 | n=0.4 | u=25 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.4 1 40 exp1-extension '[2/12]' 8 0.2

echo '▶️  正在执行任务 [3/12]: cifar-100 | sepfpl | r=8 | n=0.2 | u=25 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.2 1 40 exp1-extension '[3/12]' 8 0.2

echo '▶️  正在执行任务 [4/12]: cifar-100 | sepfpl | r=8 | n=0.1 | u=25 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.1 1 40 exp1-extension '[4/12]' 8 0.2

echo '▶️  正在执行任务 [5/12]: cifar-100 | sepfpl | r=8 | n=0.05 | u=25 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.05 1 40 exp1-extension '[5/12]' 8 0.2

echo '▶️  正在执行任务 [6/12]: cifar-100 | sepfpl | r=8 | n=0.01 | u=25 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.01 1 40 exp1-extension '[6/12]' 8 0.2

echo '▶️  正在执行任务 [7/12]: cifar-100 | sepfpl | r=8 | n=0.0 | u=50 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.0 1 40 exp1-extension '[7/12]' 8 0.2

echo '▶️  正在执行任务 [8/12]: cifar-100 | sepfpl | r=8 | n=0.4 | u=50 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.4 1 40 exp1-extension '[8/12]' 8 0.2

echo '▶️  正在执行任务 [9/12]: cifar-100 | sepfpl | r=8 | n=0.2 | u=50 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.2 1 40 exp1-extension '[9/12]' 8 0.2

echo '▶️  正在执行任务 [10/12]: cifar-100 | sepfpl | r=8 | n=0.1 | u=50 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.1 1 40 exp1-extension '[10/12]' 8 0.2

echo '▶️  正在执行任务 [11/12]: cifar-100 | sepfpl | r=8 | n=0.05 | u=50 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.05 1 40 exp1-extension '[11/12]' 8 0.2

echo '▶️  正在执行任务 [12/12]: cifar-100 | sepfpl | r=8 | n=0.01 | u=50 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.01 1 40 exp1-extension '[12/12]' 8 0.2

