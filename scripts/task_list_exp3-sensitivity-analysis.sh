#!/bin/bash

# 实验任务列表: exp3-sensitivity-analysis
# 生成时间: 2025-11-30 22:22:57
# 任务总数: 20
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/20] caltech-101 | sepfpl | r=1 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0 1 100 exp3-sensitivity-analysis '[1/20]'
    # [3/20] caltech-101 | sepfpl | r=4 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0 1 100 exp3-sensitivity-analysis '[3/20]'
    # [5/20] caltech-101 | sepfpl | r=16 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0 1 100 exp3-sensitivity-analysis '[5/20]'
    # [7/20] oxford_pets | sepfpl | r=2 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0 1 100 exp3-sensitivity-analysis '[7/20]'
    # [9/20] oxford_pets | sepfpl | r=8 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0 1 100 exp3-sensitivity-analysis '[9/20]'
    # [11/20] oxford_flowers | sepfpl | r=1 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 1 0 1 100 exp3-sensitivity-analysis '[11/20]'
    # [13/20] oxford_flowers | sepfpl | r=4 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 4 0 1 100 exp3-sensitivity-analysis '[13/20]'
    # [15/20] oxford_flowers | sepfpl | r=16 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 16 0 1 100 exp3-sensitivity-analysis '[15/20]'
    # [17/20] food-101 | sepfpl | r=2 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 2 0 1 100 exp3-sensitivity-analysis '[17/20]'
    # [19/20] food-101 | sepfpl | r=8 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0 1 100 exp3-sensitivity-analysis '[19/20]'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/20] caltech-101 | sepfpl | r=2 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0 1 100 exp3-sensitivity-analysis '[2/20]'
    # [4/20] caltech-101 | sepfpl | r=8 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0 1 100 exp3-sensitivity-analysis '[4/20]'
    # [6/20] oxford_pets | sepfpl | r=1 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0 1 100 exp3-sensitivity-analysis '[6/20]'
    # [8/20] oxford_pets | sepfpl | r=4 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0 1 100 exp3-sensitivity-analysis '[8/20]'
    # [10/20] oxford_pets | sepfpl | r=16 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0 1 100 exp3-sensitivity-analysis '[10/20]'
    # [12/20] oxford_flowers | sepfpl | r=2 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 2 0 1 100 exp3-sensitivity-analysis '[12/20]'
    # [14/20] oxford_flowers | sepfpl | r=8 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0 1 100 exp3-sensitivity-analysis '[14/20]'
    # [16/20] food-101 | sepfpl | r=1 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 1 0 1 100 exp3-sensitivity-analysis '[16/20]'
    # [18/20] food-101 | sepfpl | r=4 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 4 0 1 100 exp3-sensitivity-analysis '[18/20]'
    # [20/20] food-101 | sepfpl | r=16 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 16 0 1 100 exp3-sensitivity-analysis '[20/20]'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
