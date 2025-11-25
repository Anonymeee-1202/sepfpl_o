#!/bin/bash

# 实验任务列表: exp1-simple
# 生成时间: 2025-11-25 20:51:52
# 任务总数: 32
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 顺序执行模式 (无 GPU 或单 GPU)
echo '▶️  正在执行任务 [1/32]: caltech-101 | dpfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.0 1 5 exp1-simple '[1/32]'

echo '▶️  正在执行任务 [2/32]: caltech-101 | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 5 exp1-simple '[2/32]'

echo '▶️  正在执行任务 [3/32]: caltech-101 | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.4 1 5 exp1-simple '[3/32]'

echo '▶️  正在执行任务 [4/32]: caltech-101 | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 5 exp1-simple '[4/32]'

echo '▶️  正在执行任务 [5/32]: caltech-101 | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.1 1 5 exp1-simple '[5/32]'

echo '▶️  正在执行任务 [6/32]: caltech-101 | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 5 exp1-simple '[6/32]'

echo '▶️  正在执行任务 [7/32]: caltech-101 | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.01 1 5 exp1-simple '[7/32]'

echo '▶️  正在执行任务 [8/32]: caltech-101 | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 5 exp1-simple '[8/32]'

echo '▶️  正在执行任务 [9/32]: oxford_pets | dpfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.0 1 5 exp1-simple '[9/32]'

echo '▶️  正在执行任务 [10/32]: oxford_pets | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 1 5 exp1-simple '[10/32]'

echo '▶️  正在执行任务 [11/32]: oxford_pets | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.4 1 5 exp1-simple '[11/32]'

echo '▶️  正在执行任务 [12/32]: oxford_pets | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 5 exp1-simple '[12/32]'

echo '▶️  正在执行任务 [13/32]: oxford_pets | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.1 1 5 exp1-simple '[13/32]'

echo '▶️  正在执行任务 [14/32]: oxford_pets | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 5 exp1-simple '[14/32]'

echo '▶️  正在执行任务 [15/32]: oxford_pets | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.01 1 5 exp1-simple '[15/32]'

echo '▶️  正在执行任务 [16/32]: oxford_pets | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 5 exp1-simple '[16/32]'

echo '▶️  正在执行任务 [17/32]: oxford_flowers | dpfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.0 1 5 exp1-simple '[17/32]'

echo '▶️  正在执行任务 [18/32]: oxford_flowers | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.0 1 5 exp1-simple '[18/32]'

echo '▶️  正在执行任务 [19/32]: oxford_flowers | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.4 1 5 exp1-simple '[19/32]'

echo '▶️  正在执行任务 [20/32]: oxford_flowers | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.4 1 5 exp1-simple '[20/32]'

echo '▶️  正在执行任务 [21/32]: oxford_flowers | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.1 1 5 exp1-simple '[21/32]'

echo '▶️  正在执行任务 [22/32]: oxford_flowers | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.1 1 5 exp1-simple '[22/32]'

echo '▶️  正在执行任务 [23/32]: oxford_flowers | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.01 1 5 exp1-simple '[23/32]'

echo '▶️  正在执行任务 [24/32]: oxford_flowers | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.01 1 5 exp1-simple '[24/32]'

echo '▶️  正在执行任务 [25/32]: food-101 | dpfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.0 1 5 exp1-simple '[25/32]'

echo '▶️  正在执行任务 [26/32]: food-101 | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.0 1 5 exp1-simple '[26/32]'

echo '▶️  正在执行任务 [27/32]: food-101 | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.4 1 5 exp1-simple '[27/32]'

echo '▶️  正在执行任务 [28/32]: food-101 | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.4 1 5 exp1-simple '[28/32]'

echo '▶️  正在执行任务 [29/32]: food-101 | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.1 1 5 exp1-simple '[29/32]'

echo '▶️  正在执行任务 [30/32]: food-101 | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.1 1 5 exp1-simple '[30/32]'

echo '▶️  正在执行任务 [31/32]: food-101 | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.01 1 5 exp1-simple '[31/32]'

echo '▶️  正在执行任务 [32/32]: food-101 | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.01 1 5 exp1-simple '[32/32]'

