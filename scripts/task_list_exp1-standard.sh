#!/bin/bash

# 实验任务列表: exp1-standard
# 生成时间: 2025-12-01 18:15:04
# 任务总数: 30
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/30] stanford_dogs | promptfl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 promptfl 8 0.0 1 40 exp1-standard '[1/30]'
    # [3/30] stanford_dogs | fedpgp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedpgp 8 0.0 1 40 exp1-standard '[3/30]'
    # [5/30] stanford_dogs | sepfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 8 0.0 1 40 exp1-standard '[5/30]'
    # [7/30] stanford_dogs | fedotp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedotp 8 0.4 1 40 exp1-standard '[7/30]'
    # [9/30] stanford_dogs | dpfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 dpfpl 8 0.4 1 40 exp1-standard '[9/30]'
    # [11/30] stanford_dogs | promptfl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 promptfl 8 0.2 1 40 exp1-standard '[11/30]'
    # [13/30] stanford_dogs | fedpgp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedpgp 8 0.2 1 40 exp1-standard '[13/30]'
    # [15/30] stanford_dogs | sepfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 8 0.2 1 40 exp1-standard '[15/30]'
    # [17/30] stanford_dogs | fedotp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedotp 8 0.1 1 40 exp1-standard '[17/30]'
    # [19/30] stanford_dogs | dpfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 dpfpl 8 0.1 1 40 exp1-standard '[19/30]'
    # [21/30] stanford_dogs | promptfl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 promptfl 8 0.05 1 40 exp1-standard '[21/30]'
    # [23/30] stanford_dogs | fedpgp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedpgp 8 0.05 1 40 exp1-standard '[23/30]'
    # [25/30] stanford_dogs | sepfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 8 0.05 1 40 exp1-standard '[25/30]'
    # [27/30] stanford_dogs | fedotp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedotp 8 0.01 1 40 exp1-standard '[27/30]'
    # [29/30] stanford_dogs | dpfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 dpfpl 8 0.01 1 40 exp1-standard '[29/30]'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/30] stanford_dogs | fedotp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedotp 8 0.0 1 40 exp1-standard '[2/30]'
    # [4/30] stanford_dogs | dpfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 dpfpl 8 0.0 1 40 exp1-standard '[4/30]'
    # [6/30] stanford_dogs | promptfl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 promptfl 8 0.4 1 40 exp1-standard '[6/30]'
    # [8/30] stanford_dogs | fedpgp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedpgp 8 0.4 1 40 exp1-standard '[8/30]'
    # [10/30] stanford_dogs | sepfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 8 0.4 1 40 exp1-standard '[10/30]'
    # [12/30] stanford_dogs | fedotp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedotp 8 0.2 1 40 exp1-standard '[12/30]'
    # [14/30] stanford_dogs | dpfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 dpfpl 8 0.2 1 40 exp1-standard '[14/30]'
    # [16/30] stanford_dogs | promptfl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 promptfl 8 0.1 1 40 exp1-standard '[16/30]'
    # [18/30] stanford_dogs | fedpgp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedpgp 8 0.1 1 40 exp1-standard '[18/30]'
    # [20/30] stanford_dogs | sepfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 8 0.1 1 40 exp1-standard '[20/30]'
    # [22/30] stanford_dogs | fedotp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedotp 8 0.05 1 40 exp1-standard '[22/30]'
    # [24/30] stanford_dogs | dpfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 dpfpl 8 0.05 1 40 exp1-standard '[24/30]'
    # [26/30] stanford_dogs | promptfl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 promptfl 8 0.01 1 40 exp1-standard '[26/30]'
    # [28/30] stanford_dogs | fedpgp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 fedpgp 8 0.01 1 40 exp1-standard '[28/30]'
    # [30/30] stanford_dogs | sepfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/stanford_dogs.yaml 10 sepfpl 8 0.01 1 40 exp1-standard '[30/30]'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
