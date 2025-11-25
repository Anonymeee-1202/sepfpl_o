#!/bin/bash

# 实验任务列表: exp1-simple
# 生成时间: 2025-11-25 16:17:16
# 任务总数: 16
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 顺序执行模式 (无 GPU 或单 GPU)
echo '▶️  正在执行任务 [1/16]: caltech-101 | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 50 exp1-simple '[1/16]'

echo '▶️  正在执行任务 [2/16]: caltech-101 | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 50 exp1-simple '[2/16]'

echo '▶️  正在执行任务 [3/16]: caltech-101 | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 50 exp1-simple '[3/16]'

echo '▶️  正在执行任务 [4/16]: caltech-101 | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 50 exp1-simple '[4/16]'

echo '▶️  正在执行任务 [5/16]: oxford_pets | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 1 50 exp1-simple '[5/16]'

echo '▶️  正在执行任务 [6/16]: oxford_pets | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 50 exp1-simple '[6/16]'

echo '▶️  正在执行任务 [7/16]: oxford_pets | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 50 exp1-simple '[7/16]'

echo '▶️  正在执行任务 [8/16]: oxford_pets | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 50 exp1-simple '[8/16]'

echo '▶️  正在执行任务 [9/16]: oxford_flowers | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.0 1 50 exp1-simple '[9/16]'

echo '▶️  正在执行任务 [10/16]: oxford_flowers | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.4 1 50 exp1-simple '[10/16]'

echo '▶️  正在执行任务 [11/16]: oxford_flowers | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.1 1 50 exp1-simple '[11/16]'

echo '▶️  正在执行任务 [12/16]: oxford_flowers | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.01 1 50 exp1-simple '[12/16]'

echo '▶️  正在执行任务 [13/16]: food-101 | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.0 1 50 exp1-simple '[13/16]'

echo '▶️  正在执行任务 [14/16]: food-101 | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.4 1 50 exp1-simple '[14/16]'

echo '▶️  正在执行任务 [15/16]: food-101 | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.1 1 50 exp1-simple '[15/16]'

echo '▶️  正在执行任务 [16/16]: food-101 | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.01 1 50 exp1-simple '[16/16]'

