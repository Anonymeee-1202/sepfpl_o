#!/bin/bash

# 实验任务列表: exp1-hard
# 生成时间: 2025-11-27 16:40:36
# 任务总数: 60
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/60] cifar-100 | promptfl | r=8 n=0.0 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.0 1 40 exp1-hard '[1/60]'
    # [3/60] cifar-100 | fedpgp | r=8 n=0.0 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.0 1 40 exp1-hard '[3/60]'
    # [5/60] cifar-100 | sepfpl | r=8 n=0.0 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.0 1 40 exp1-hard '[5/60]'
    # [7/60] cifar-100 | fedotp | r=8 n=0.4 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.4 1 40 exp1-hard '[7/60]'
    # [9/60] cifar-100 | dpfpl | r=8 n=0.4 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.4 1 40 exp1-hard '[9/60]'
    # [11/60] cifar-100 | promptfl | r=8 n=0.2 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.2 1 40 exp1-hard '[11/60]'
    # [13/60] cifar-100 | fedpgp | r=8 n=0.2 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.2 1 40 exp1-hard '[13/60]'
    # [15/60] cifar-100 | sepfpl | r=8 n=0.2 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.2 1 40 exp1-hard '[15/60]'
    # [17/60] cifar-100 | fedotp | r=8 n=0.1 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.1 1 40 exp1-hard '[17/60]'
    # [19/60] cifar-100 | dpfpl | r=8 n=0.1 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.1 1 40 exp1-hard '[19/60]'
    # [21/60] cifar-100 | promptfl | r=8 n=0.05 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.05 1 40 exp1-hard '[21/60]'
    # [23/60] cifar-100 | fedpgp | r=8 n=0.05 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.05 1 40 exp1-hard '[23/60]'
    # [25/60] cifar-100 | sepfpl | r=8 n=0.05 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.05 1 40 exp1-hard '[25/60]'
    # [27/60] cifar-100 | fedotp | r=8 n=0.01 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.01 1 40 exp1-hard '[27/60]'
    # [29/60] cifar-100 | dpfpl | r=8 n=0.01 u=25 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.01 1 40 exp1-hard '[29/60]'
    # [31/60] cifar-100 | promptfl | r=8 n=0.0 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.0 1 40 exp1-hard '[31/60]'
    # [33/60] cifar-100 | fedpgp | r=8 n=0.0 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.0 1 40 exp1-hard '[33/60]'
    # [35/60] cifar-100 | sepfpl | r=8 n=0.0 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.0 1 40 exp1-hard '[35/60]'
    # [37/60] cifar-100 | fedotp | r=8 n=0.4 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.4 1 40 exp1-hard '[37/60]'
    # [39/60] cifar-100 | dpfpl | r=8 n=0.4 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.4 1 40 exp1-hard '[39/60]'
    # [41/60] cifar-100 | promptfl | r=8 n=0.2 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.2 1 40 exp1-hard '[41/60]'
    # [43/60] cifar-100 | fedpgp | r=8 n=0.2 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.2 1 40 exp1-hard '[43/60]'
    # [45/60] cifar-100 | sepfpl | r=8 n=0.2 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.2 1 40 exp1-hard '[45/60]'
    # [47/60] cifar-100 | fedotp | r=8 n=0.1 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.1 1 40 exp1-hard '[47/60]'
    # [49/60] cifar-100 | dpfpl | r=8 n=0.1 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.1 1 40 exp1-hard '[49/60]'
    # [51/60] cifar-100 | promptfl | r=8 n=0.05 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.05 1 40 exp1-hard '[51/60]'
    # [53/60] cifar-100 | fedpgp | r=8 n=0.05 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.05 1 40 exp1-hard '[53/60]'
    # [55/60] cifar-100 | sepfpl | r=8 n=0.05 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.05 1 40 exp1-hard '[55/60]'
    # [57/60] cifar-100 | fedotp | r=8 n=0.01 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.01 1 40 exp1-hard '[57/60]'
    # [59/60] cifar-100 | dpfpl | r=8 n=0.01 u=50 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.01 1 40 exp1-hard '[59/60]'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/60] cifar-100 | fedotp | r=8 n=0.0 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.0 1 40 exp1-hard '[2/60]'
    # [4/60] cifar-100 | dpfpl | r=8 n=0.0 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.0 1 40 exp1-hard '[4/60]'
    # [6/60] cifar-100 | promptfl | r=8 n=0.4 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.4 1 40 exp1-hard '[6/60]'
    # [8/60] cifar-100 | fedpgp | r=8 n=0.4 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.4 1 40 exp1-hard '[8/60]'
    # [10/60] cifar-100 | sepfpl | r=8 n=0.4 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.4 1 40 exp1-hard '[10/60]'
    # [12/60] cifar-100 | fedotp | r=8 n=0.2 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.2 1 40 exp1-hard '[12/60]'
    # [14/60] cifar-100 | dpfpl | r=8 n=0.2 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.2 1 40 exp1-hard '[14/60]'
    # [16/60] cifar-100 | promptfl | r=8 n=0.1 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.1 1 40 exp1-hard '[16/60]'
    # [18/60] cifar-100 | fedpgp | r=8 n=0.1 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.1 1 40 exp1-hard '[18/60]'
    # [20/60] cifar-100 | sepfpl | r=8 n=0.1 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.1 1 40 exp1-hard '[20/60]'
    # [22/60] cifar-100 | fedotp | r=8 n=0.05 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.05 1 40 exp1-hard '[22/60]'
    # [24/60] cifar-100 | dpfpl | r=8 n=0.05 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.05 1 40 exp1-hard '[24/60]'
    # [26/60] cifar-100 | promptfl | r=8 n=0.01 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.01 1 40 exp1-hard '[26/60]'
    # [28/60] cifar-100 | fedpgp | r=8 n=0.01 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.01 1 40 exp1-hard '[28/60]'
    # [30/60] cifar-100 | sepfpl | r=8 n=0.01 u=25 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.01 1 40 exp1-hard '[30/60]'
    # [32/60] cifar-100 | fedotp | r=8 n=0.0 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.0 1 40 exp1-hard '[32/60]'
    # [34/60] cifar-100 | dpfpl | r=8 n=0.0 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.0 1 40 exp1-hard '[34/60]'
    # [36/60] cifar-100 | promptfl | r=8 n=0.4 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.4 1 40 exp1-hard '[36/60]'
    # [38/60] cifar-100 | fedpgp | r=8 n=0.4 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.4 1 40 exp1-hard '[38/60]'
    # [40/60] cifar-100 | sepfpl | r=8 n=0.4 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.4 1 40 exp1-hard '[40/60]'
    # [42/60] cifar-100 | fedotp | r=8 n=0.2 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.2 1 40 exp1-hard '[42/60]'
    # [44/60] cifar-100 | dpfpl | r=8 n=0.2 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.2 1 40 exp1-hard '[44/60]'
    # [46/60] cifar-100 | promptfl | r=8 n=0.1 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.1 1 40 exp1-hard '[46/60]'
    # [48/60] cifar-100 | fedpgp | r=8 n=0.1 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.1 1 40 exp1-hard '[48/60]'
    # [50/60] cifar-100 | sepfpl | r=8 n=0.1 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.1 1 40 exp1-hard '[50/60]'
    # [52/60] cifar-100 | fedotp | r=8 n=0.05 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.05 1 40 exp1-hard '[52/60]'
    # [54/60] cifar-100 | dpfpl | r=8 n=0.05 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.05 1 40 exp1-hard '[54/60]'
    # [56/60] cifar-100 | promptfl | r=8 n=0.01 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.01 1 40 exp1-hard '[56/60]'
    # [58/60] cifar-100 | fedpgp | r=8 n=0.01 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.01 1 40 exp1-hard '[58/60]'
    # [60/60] cifar-100 | sepfpl | r=8 n=0.01 u=50 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.01 1 40 exp1-hard '[60/60]'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
