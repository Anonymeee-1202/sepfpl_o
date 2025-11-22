#!/bin/bash

# 实验任务列表: exp2
# 生成时间: 2025-11-21 17:54:47
# 任务总数: 60
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/60] caltech-101 | sepfpl_hcse | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 1 0.4 1 30 exp2 '[1/60]'
    # [3/60] caltech-101 | sepfpl_hcse | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 2 0.4 1 30 exp2 '[3/60]'
    # [5/60] caltech-101 | sepfpl_hcse | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 4 0.4 1 30 exp2 '[5/60]'
    # [7/60] caltech-101 | sepfpl_hcse | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.4 1 30 exp2 '[7/60]'
    # [9/60] caltech-101 | sepfpl_hcse | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 16 0.4 1 30 exp2 '[9/60]'
    # [11/60] caltech-101 | sepfpl_hcse | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 1 0.1 1 30 exp2 '[11/60]'
    # [13/60] caltech-101 | sepfpl_hcse | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 2 0.1 1 30 exp2 '[13/60]'
    # [15/60] caltech-101 | sepfpl_hcse | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 4 0.1 1 30 exp2 '[15/60]'
    # [17/60] caltech-101 | sepfpl_hcse | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.1 1 30 exp2 '[17/60]'
    # [19/60] caltech-101 | sepfpl_hcse | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 16 0.1 1 30 exp2 '[19/60]'
    # [21/60] caltech-101 | sepfpl_hcse | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 1 0.01 1 30 exp2 '[21/60]'
    # [23/60] caltech-101 | sepfpl_hcse | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 2 0.01 1 30 exp2 '[23/60]'
    # [25/60] caltech-101 | sepfpl_hcse | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 4 0.01 1 30 exp2 '[25/60]'
    # [27/60] caltech-101 | sepfpl_hcse | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.01 1 30 exp2 '[27/60]'
    # [29/60] caltech-101 | sepfpl_hcse | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 16 0.01 1 30 exp2 '[29/60]'
    # [31/60] oxford_pets | sepfpl_hcse | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 1 0.4 1 30 exp2 '[31/60]'
    # [33/60] oxford_pets | sepfpl_hcse | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 2 0.4 1 30 exp2 '[33/60]'
    # [35/60] oxford_pets | sepfpl_hcse | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 4 0.4 1 30 exp2 '[35/60]'
    # [37/60] oxford_pets | sepfpl_hcse | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.4 1 30 exp2 '[37/60]'
    # [39/60] oxford_pets | sepfpl_hcse | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 16 0.4 1 30 exp2 '[39/60]'
    # [41/60] oxford_pets | sepfpl_hcse | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 1 0.1 1 30 exp2 '[41/60]'
    # [43/60] oxford_pets | sepfpl_hcse | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 2 0.1 1 30 exp2 '[43/60]'
    # [45/60] oxford_pets | sepfpl_hcse | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 4 0.1 1 30 exp2 '[45/60]'
    # [47/60] oxford_pets | sepfpl_hcse | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.1 1 30 exp2 '[47/60]'
    # [49/60] oxford_pets | sepfpl_hcse | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 16 0.1 1 30 exp2 '[49/60]'
    # [51/60] oxford_pets | sepfpl_hcse | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 1 0.01 1 30 exp2 '[51/60]'
    # [53/60] oxford_pets | sepfpl_hcse | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 2 0.01 1 30 exp2 '[53/60]'
    # [55/60] oxford_pets | sepfpl_hcse | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 4 0.01 1 30 exp2 '[55/60]'
    # [57/60] oxford_pets | sepfpl_hcse | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.01 1 30 exp2 '[57/60]'
    # [59/60] oxford_pets | sepfpl_hcse | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 16 0.01 1 30 exp2 '[59/60]'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/60] caltech-101 | sepfpl_time_adaptive | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.4 1 30 exp2 '[2/60]'
    # [4/60] caltech-101 | sepfpl_time_adaptive | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.4 1 30 exp2 '[4/60]'
    # [6/60] caltech-101 | sepfpl_time_adaptive | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.4 1 30 exp2 '[6/60]'
    # [8/60] caltech-101 | sepfpl_time_adaptive | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.4 1 30 exp2 '[8/60]'
    # [10/60] caltech-101 | sepfpl_time_adaptive | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.4 1 30 exp2 '[10/60]'
    # [12/60] caltech-101 | sepfpl_time_adaptive | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.1 1 30 exp2 '[12/60]'
    # [14/60] caltech-101 | sepfpl_time_adaptive | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.1 1 30 exp2 '[14/60]'
    # [16/60] caltech-101 | sepfpl_time_adaptive | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.1 1 30 exp2 '[16/60]'
    # [18/60] caltech-101 | sepfpl_time_adaptive | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 exp2 '[18/60]'
    # [20/60] caltech-101 | sepfpl_time_adaptive | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.1 1 30 exp2 '[20/60]'
    # [22/60] caltech-101 | sepfpl_time_adaptive | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.01 1 30 exp2 '[22/60]'
    # [24/60] caltech-101 | sepfpl_time_adaptive | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.01 1 30 exp2 '[24/60]'
    # [26/60] caltech-101 | sepfpl_time_adaptive | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.01 1 30 exp2 '[26/60]'
    # [28/60] caltech-101 | sepfpl_time_adaptive | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.01 1 30 exp2 '[28/60]'
    # [30/60] caltech-101 | sepfpl_time_adaptive | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.01 1 30 exp2 '[30/60]'
    # [32/60] oxford_pets | sepfpl_time_adaptive | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.4 1 30 exp2 '[32/60]'
    # [34/60] oxford_pets | sepfpl_time_adaptive | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.4 1 30 exp2 '[34/60]'
    # [36/60] oxford_pets | sepfpl_time_adaptive | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.4 1 30 exp2 '[36/60]'
    # [38/60] oxford_pets | sepfpl_time_adaptive | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.4 1 30 exp2 '[38/60]'
    # [40/60] oxford_pets | sepfpl_time_adaptive | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.4 1 30 exp2 '[40/60]'
    # [42/60] oxford_pets | sepfpl_time_adaptive | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.1 1 30 exp2 '[42/60]'
    # [44/60] oxford_pets | sepfpl_time_adaptive | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.1 1 30 exp2 '[44/60]'
    # [46/60] oxford_pets | sepfpl_time_adaptive | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.1 1 30 exp2 '[46/60]'
    # [48/60] oxford_pets | sepfpl_time_adaptive | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 exp2 '[48/60]'
    # [50/60] oxford_pets | sepfpl_time_adaptive | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.1 1 30 exp2 '[50/60]'
    # [52/60] oxford_pets | sepfpl_time_adaptive | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.01 1 30 exp2 '[52/60]'
    # [54/60] oxford_pets | sepfpl_time_adaptive | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.01 1 30 exp2 '[54/60]'
    # [56/60] oxford_pets | sepfpl_time_adaptive | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.01 1 30 exp2 '[56/60]'
    # [58/60] oxford_pets | sepfpl_time_adaptive | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.01 1 30 exp2 '[58/60]'
    # [60/60] oxford_pets | sepfpl_time_adaptive | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.01 1 30 exp2 '[60/60]'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
