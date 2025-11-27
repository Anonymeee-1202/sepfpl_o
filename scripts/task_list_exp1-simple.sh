#!/bin/bash

# 实验任务列表: exp1-simple
# 生成时间: 2025-11-27 16:40:36
# 任务总数: 4
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/4] oxford_flowers | promptfl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.01 1 40 exp1-simple '[1/4]'
    # [3/4] oxford_flowers | fedpgp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.01 1 40 exp1-simple '[3/4]'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/4] oxford_flowers | fedotp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.01 1 40 exp1-simple '[2/4]'
    # [4/4] oxford_flowers | sepfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.01 1 40 exp1-simple '[4/4]'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
