#!/bin/bash

# 实验任务列表: exp1-extension
# 生成时间: 2025-12-01 19:25:13
# 任务总数: 12
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/12] cifar-100 | sepfpl | r=8 n=0.0 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.0 1 40 exp1-extension '[1/12]'
    # [3/12] cifar-100 | sepfpl | r=8 n=0.2 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.2 1 40 exp1-extension '[3/12]'
    # [5/12] cifar-100 | sepfpl | r=8 n=0.05 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.05 1 40 exp1-extension '[5/12]'
    # [7/12] cifar-100 | sepfpl | r=8 n=0.0 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.0 1 40 exp1-extension '[7/12]'
    # [9/12] cifar-100 | sepfpl | r=8 n=0.2 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.2 1 40 exp1-extension '[9/12]'
    # [11/12] cifar-100 | sepfpl | r=8 n=0.05 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.05 1 40 exp1-extension '[11/12]'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/12] cifar-100 | sepfpl | r=8 n=0.4 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.4 1 40 exp1-extension '[2/12]'
    # [4/12] cifar-100 | sepfpl | r=8 n=0.1 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.1 1 40 exp1-extension '[4/12]'
    # [6/12] cifar-100 | sepfpl | r=8 n=0.01 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.01 1 40 exp1-extension '[6/12]'
    # [8/12] cifar-100 | sepfpl | r=8 n=0.4 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.4 1 40 exp1-extension '[8/12]'
    # [10/12] cifar-100 | sepfpl | r=8 n=0.1 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.1 1 40 exp1-extension '[10/12]'
    # [12/12] cifar-100 | sepfpl | r=8 n=0.01 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.01 1 40 exp1-extension '[12/12]'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
