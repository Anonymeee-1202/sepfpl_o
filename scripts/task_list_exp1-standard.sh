#!/bin/bash

# 实验任务列表: exp1-standard
# 生成时间: 2025-12-05 17:17:47
# 任务总数: 12
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 顺序执行模式 (无 GPU 或单 GPU)
echo '▶️  正在执行任务 [1/12]: caltech-101 | sepfpl | r=8 | n=0.0 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 40 exp1-standard '[1/12]' 8 0.2

echo '▶️  正在执行任务 [2/12]: caltech-101 | sepfpl | r=8 | n=0.2 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.2 1 40 exp1-standard '[2/12]' 8 0.2

echo '▶️  正在执行任务 [3/12]: caltech-101 | sepfpl | r=8 | n=0.05 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.05 1 40 exp1-standard '[3/12]' 8 0.2

echo '▶️  正在执行任务 [4/12]: oxford_flowers | sepfpl | r=8 | n=0.0 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.0 1 40 exp1-standard '[4/12]' 8 0.2

echo '▶️  正在执行任务 [5/12]: oxford_flowers | sepfpl | r=8 | n=0.2 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.2 1 40 exp1-standard '[5/12]' 8 0.2

echo '▶️  正在执行任务 [6/12]: oxford_flowers | sepfpl | r=8 | n=0.05 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.05 1 40 exp1-standard '[6/12]' 8 0.2

echo '▶️  正在执行任务 [7/12]: food-101 | sepfpl | r=8 | n=0.0 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.0 1 40 exp1-standard '[7/12]' 8 0.2

echo '▶️  正在执行任务 [8/12]: food-101 | sepfpl | r=8 | n=0.2 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.2 1 40 exp1-standard '[8/12]' 8 0.2

echo '▶️  正在执行任务 [9/12]: food-101 | sepfpl | r=8 | n=0.05 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.05 1 40 exp1-standard '[9/12]' 8 0.2

echo '▶️  正在执行任务 [10/12]: stanford_dogs | sepfpl | r=8 | n=0.0 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 8 0.0 1 40 exp1-standard '[10/12]' 8 0.2

echo '▶️  正在执行任务 [11/12]: stanford_dogs | sepfpl | r=8 | n=0.2 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 8 0.2 1 40 exp1-standard '[11/12]' 8 0.2

echo '▶️  正在执行任务 [12/12]: stanford_dogs | sepfpl | r=8 | n=0.05 | u=10 | s=1'
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 8 0.05 1 40 exp1-standard '[12/12]' 8 0.2

