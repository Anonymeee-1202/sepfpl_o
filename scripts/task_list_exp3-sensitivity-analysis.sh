#!/bin/bash

# 实验任务列表: exp3-sensitivity-analysis
# 生成时间: 2025-12-02 00:18:40
# 任务总数: 64
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/64] caltech-101 | sepfpl | r=1 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0 1 40 exp3-sensitivity-analysis '[1/64]'
    # [3/64] caltech-101 | sepfpl | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.1 1 40 exp3-sensitivity-analysis '[3/64]'
    # [5/64] caltech-101 | sepfpl | r=2 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0 1 40 exp3-sensitivity-analysis '[5/64]'
    # [7/64] caltech-101 | sepfpl | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.1 1 40 exp3-sensitivity-analysis '[7/64]'
    # [9/64] caltech-101 | sepfpl | r=4 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0 1 40 exp3-sensitivity-analysis '[9/64]'
    # [11/64] caltech-101 | sepfpl | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.1 1 40 exp3-sensitivity-analysis '[11/64]'
    # [13/64] caltech-101 | sepfpl | r=16 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0 1 40 exp3-sensitivity-analysis '[13/64]'
    # [15/64] caltech-101 | sepfpl | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.1 1 40 exp3-sensitivity-analysis '[15/64]'
    # [17/64] stanford_dogs | sepfpl | r=1 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 1 0 1 40 exp3-sensitivity-analysis '[17/64]'
    # [19/64] stanford_dogs | sepfpl | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 1 0.1 1 40 exp3-sensitivity-analysis '[19/64]'
    # [21/64] stanford_dogs | sepfpl | r=2 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 2 0 1 40 exp3-sensitivity-analysis '[21/64]'
    # [23/64] stanford_dogs | sepfpl | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 2 0.1 1 40 exp3-sensitivity-analysis '[23/64]'
    # [25/64] stanford_dogs | sepfpl | r=4 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 4 0 1 40 exp3-sensitivity-analysis '[25/64]'
    # [27/64] stanford_dogs | sepfpl | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 4 0.1 1 40 exp3-sensitivity-analysis '[27/64]'
    # [29/64] stanford_dogs | sepfpl | r=16 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 16 0 1 40 exp3-sensitivity-analysis '[29/64]'
    # [31/64] stanford_dogs | sepfpl | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 16 0.1 1 40 exp3-sensitivity-analysis '[31/64]'
    # [33/64] oxford_flowers | sepfpl | r=1 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 1 0 1 40 exp3-sensitivity-analysis '[33/64]'
    # [35/64] oxford_flowers | sepfpl | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 1 0.1 1 40 exp3-sensitivity-analysis '[35/64]'
    # [37/64] oxford_flowers | sepfpl | r=2 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 2 0 1 40 exp3-sensitivity-analysis '[37/64]'
    # [39/64] oxford_flowers | sepfpl | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 2 0.1 1 40 exp3-sensitivity-analysis '[39/64]'
    # [41/64] oxford_flowers | sepfpl | r=4 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 4 0 1 40 exp3-sensitivity-analysis '[41/64]'
    # [43/64] oxford_flowers | sepfpl | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 4 0.1 1 40 exp3-sensitivity-analysis '[43/64]'
    # [45/64] oxford_flowers | sepfpl | r=16 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 16 0 1 40 exp3-sensitivity-analysis '[45/64]'
    # [47/64] oxford_flowers | sepfpl | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 16 0.1 1 40 exp3-sensitivity-analysis '[47/64]'
    # [49/64] food-101 | sepfpl | r=1 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 1 0 1 40 exp3-sensitivity-analysis '[49/64]'
    # [51/64] food-101 | sepfpl | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 1 0.1 1 40 exp3-sensitivity-analysis '[51/64]'
    # [53/64] food-101 | sepfpl | r=2 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 2 0 1 40 exp3-sensitivity-analysis '[53/64]'
    # [55/64] food-101 | sepfpl | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 2 0.1 1 40 exp3-sensitivity-analysis '[55/64]'
    # [57/64] food-101 | sepfpl | r=4 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 4 0 1 40 exp3-sensitivity-analysis '[57/64]'
    # [59/64] food-101 | sepfpl | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 4 0.1 1 40 exp3-sensitivity-analysis '[59/64]'
    # [61/64] food-101 | sepfpl | r=16 n=0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 16 0 1 40 exp3-sensitivity-analysis '[61/64]'
    # [63/64] food-101 | sepfpl | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 16 0.1 1 40 exp3-sensitivity-analysis '[63/64]'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/64] caltech-101 | sepfpl | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.4 1 40 exp3-sensitivity-analysis '[2/64]'
    # [4/64] caltech-101 | sepfpl | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.01 1 40 exp3-sensitivity-analysis '[4/64]'
    # [6/64] caltech-101 | sepfpl | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.4 1 40 exp3-sensitivity-analysis '[6/64]'
    # [8/64] caltech-101 | sepfpl | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.01 1 40 exp3-sensitivity-analysis '[8/64]'
    # [10/64] caltech-101 | sepfpl | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.4 1 40 exp3-sensitivity-analysis '[10/64]'
    # [12/64] caltech-101 | sepfpl | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.01 1 40 exp3-sensitivity-analysis '[12/64]'
    # [14/64] caltech-101 | sepfpl | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.4 1 40 exp3-sensitivity-analysis '[14/64]'
    # [16/64] caltech-101 | sepfpl | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.01 1 40 exp3-sensitivity-analysis '[16/64]'
    # [18/64] stanford_dogs | sepfpl | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 1 0.4 1 40 exp3-sensitivity-analysis '[18/64]'
    # [20/64] stanford_dogs | sepfpl | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 1 0.01 1 40 exp3-sensitivity-analysis '[20/64]'
    # [22/64] stanford_dogs | sepfpl | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 2 0.4 1 40 exp3-sensitivity-analysis '[22/64]'
    # [24/64] stanford_dogs | sepfpl | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 2 0.01 1 40 exp3-sensitivity-analysis '[24/64]'
    # [26/64] stanford_dogs | sepfpl | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 4 0.4 1 40 exp3-sensitivity-analysis '[26/64]'
    # [28/64] stanford_dogs | sepfpl | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 4 0.01 1 40 exp3-sensitivity-analysis '[28/64]'
    # [30/64] stanford_dogs | sepfpl | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 16 0.4 1 40 exp3-sensitivity-analysis '[30/64]'
    # [32/64] stanford_dogs | sepfpl | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 16 0.01 1 40 exp3-sensitivity-analysis '[32/64]'
    # [34/64] oxford_flowers | sepfpl | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 1 0.4 1 40 exp3-sensitivity-analysis '[34/64]'
    # [36/64] oxford_flowers | sepfpl | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 1 0.01 1 40 exp3-sensitivity-analysis '[36/64]'
    # [38/64] oxford_flowers | sepfpl | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 2 0.4 1 40 exp3-sensitivity-analysis '[38/64]'
    # [40/64] oxford_flowers | sepfpl | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 2 0.01 1 40 exp3-sensitivity-analysis '[40/64]'
    # [42/64] oxford_flowers | sepfpl | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 4 0.4 1 40 exp3-sensitivity-analysis '[42/64]'
    # [44/64] oxford_flowers | sepfpl | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 4 0.01 1 40 exp3-sensitivity-analysis '[44/64]'
    # [46/64] oxford_flowers | sepfpl | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 16 0.4 1 40 exp3-sensitivity-analysis '[46/64]'
    # [48/64] oxford_flowers | sepfpl | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 16 0.01 1 40 exp3-sensitivity-analysis '[48/64]'
    # [50/64] food-101 | sepfpl | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 1 0.4 1 40 exp3-sensitivity-analysis '[50/64]'
    # [52/64] food-101 | sepfpl | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 1 0.01 1 40 exp3-sensitivity-analysis '[52/64]'
    # [54/64] food-101 | sepfpl | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 2 0.4 1 40 exp3-sensitivity-analysis '[54/64]'
    # [56/64] food-101 | sepfpl | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 2 0.01 1 40 exp3-sensitivity-analysis '[56/64]'
    # [58/64] food-101 | sepfpl | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 4 0.4 1 40 exp3-sensitivity-analysis '[58/64]'
    # [60/64] food-101 | sepfpl | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 4 0.01 1 40 exp3-sensitivity-analysis '[60/64]'
    # [62/64] food-101 | sepfpl | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 16 0.4 1 40 exp3-sensitivity-analysis '[62/64]'
    # [64/64] food-101 | sepfpl | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 16 0.01 1 40 exp3-sensitivity-analysis '[64/64]'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
