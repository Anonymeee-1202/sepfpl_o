#!/bin/bash

# 实验任务列表: exp2
# 生成时间: 2025-11-25 11:47:27
# 任务总数: 120
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/120] caltech-101 | dpfpl | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 1 0.4 1 30 exp2 '[1/120]'
    # [3/120] caltech-101 | sepfpl_time_adaptive | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.4 1 30 exp2 '[3/120]'
    # [5/120] caltech-101 | dpfpl | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 1 0.1 1 30 exp2 '[5/120]'
    # [7/120] caltech-101 | sepfpl_time_adaptive | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.1 1 30 exp2 '[7/120]'
    # [9/120] caltech-101 | dpfpl | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 1 0.01 1 30 exp2 '[9/120]'
    # [11/120] caltech-101 | sepfpl_time_adaptive | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.01 1 30 exp2 '[11/120]'
    # [13/120] caltech-101 | dpfpl | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 2 0.4 1 30 exp2 '[13/120]'
    # [15/120] caltech-101 | sepfpl_time_adaptive | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.4 1 30 exp2 '[15/120]'
    # [17/120] caltech-101 | dpfpl | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 2 0.1 1 30 exp2 '[17/120]'
    # [19/120] caltech-101 | sepfpl_time_adaptive | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.1 1 30 exp2 '[19/120]'
    # [21/120] caltech-101 | dpfpl | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 2 0.01 1 30 exp2 '[21/120]'
    # [23/120] caltech-101 | sepfpl_time_adaptive | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.01 1 30 exp2 '[23/120]'
    # [25/120] caltech-101 | dpfpl | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 4 0.4 1 30 exp2 '[25/120]'
    # [27/120] caltech-101 | sepfpl_time_adaptive | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.4 1 30 exp2 '[27/120]'
    # [29/120] caltech-101 | dpfpl | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 4 0.1 1 30 exp2 '[29/120]'
    # [31/120] caltech-101 | sepfpl_time_adaptive | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.1 1 30 exp2 '[31/120]'
    # [33/120] caltech-101 | dpfpl | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 4 0.01 1 30 exp2 '[33/120]'
    # [35/120] caltech-101 | sepfpl_time_adaptive | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.01 1 30 exp2 '[35/120]'
    # [37/120] caltech-101 | dpfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.4 1 30 exp2 '[37/120]'
    # [39/120] caltech-101 | sepfpl_time_adaptive | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.4 1 30 exp2 '[39/120]'
    # [41/120] caltech-101 | dpfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.1 1 30 exp2 '[41/120]'
    # [43/120] caltech-101 | sepfpl_time_adaptive | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 exp2 '[43/120]'
    # [45/120] caltech-101 | dpfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.01 1 30 exp2 '[45/120]'
    # [47/120] caltech-101 | sepfpl_time_adaptive | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.01 1 30 exp2 '[47/120]'
    # [49/120] caltech-101 | dpfpl | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 16 0.4 1 30 exp2 '[49/120]'
    # [51/120] caltech-101 | sepfpl_time_adaptive | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.4 1 30 exp2 '[51/120]'
    # [53/120] caltech-101 | dpfpl | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 16 0.1 1 30 exp2 '[53/120]'
    # [55/120] caltech-101 | sepfpl_time_adaptive | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.1 1 30 exp2 '[55/120]'
    # [57/120] caltech-101 | dpfpl | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 16 0.01 1 30 exp2 '[57/120]'
    # [59/120] caltech-101 | sepfpl_time_adaptive | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.01 1 30 exp2 '[59/120]'
    # [61/120] oxford_pets | dpfpl | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 1 0.4 1 30 exp2 '[61/120]'
    # [63/120] oxford_pets | sepfpl_time_adaptive | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.4 1 30 exp2 '[63/120]'
    # [65/120] oxford_pets | dpfpl | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 1 0.1 1 30 exp2 '[65/120]'
    # [67/120] oxford_pets | sepfpl_time_adaptive | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.1 1 30 exp2 '[67/120]'
    # [69/120] oxford_pets | dpfpl | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 1 0.01 1 30 exp2 '[69/120]'
    # [71/120] oxford_pets | sepfpl_time_adaptive | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.01 1 30 exp2 '[71/120]'
    # [73/120] oxford_pets | dpfpl | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 2 0.4 1 30 exp2 '[73/120]'
    # [75/120] oxford_pets | sepfpl_time_adaptive | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.4 1 30 exp2 '[75/120]'
    # [77/120] oxford_pets | dpfpl | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 2 0.1 1 30 exp2 '[77/120]'
    # [79/120] oxford_pets | sepfpl_time_adaptive | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.1 1 30 exp2 '[79/120]'
    # [81/120] oxford_pets | dpfpl | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 2 0.01 1 30 exp2 '[81/120]'
    # [83/120] oxford_pets | sepfpl_time_adaptive | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.01 1 30 exp2 '[83/120]'
    # [85/120] oxford_pets | dpfpl | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 4 0.4 1 30 exp2 '[85/120]'
    # [87/120] oxford_pets | sepfpl_time_adaptive | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.4 1 30 exp2 '[87/120]'
    # [89/120] oxford_pets | dpfpl | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 4 0.1 1 30 exp2 '[89/120]'
    # [91/120] oxford_pets | sepfpl_time_adaptive | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.1 1 30 exp2 '[91/120]'
    # [93/120] oxford_pets | dpfpl | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 4 0.01 1 30 exp2 '[93/120]'
    # [95/120] oxford_pets | sepfpl_time_adaptive | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.01 1 30 exp2 '[95/120]'
    # [97/120] oxford_pets | dpfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.4 1 30 exp2 '[97/120]'
    # [99/120] oxford_pets | sepfpl_time_adaptive | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.4 1 30 exp2 '[99/120]'
    # [101/120] oxford_pets | dpfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.1 1 30 exp2 '[101/120]'
    # [103/120] oxford_pets | sepfpl_time_adaptive | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 exp2 '[103/120]'
    # [105/120] oxford_pets | dpfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.01 1 30 exp2 '[105/120]'
    # [107/120] oxford_pets | sepfpl_time_adaptive | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.01 1 30 exp2 '[107/120]'
    # [109/120] oxford_pets | dpfpl | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 16 0.4 1 30 exp2 '[109/120]'
    # [111/120] oxford_pets | sepfpl_time_adaptive | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.4 1 30 exp2 '[111/120]'
    # [113/120] oxford_pets | dpfpl | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 16 0.1 1 30 exp2 '[113/120]'
    # [115/120] oxford_pets | sepfpl_time_adaptive | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.1 1 30 exp2 '[115/120]'
    # [117/120] oxford_pets | dpfpl | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 16 0.01 1 30 exp2 '[117/120]'
    # [119/120] oxford_pets | sepfpl_time_adaptive | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.01 1 30 exp2 '[119/120]'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/120] caltech-101 | sepfpl_hcse | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 1 0.4 1 30 exp2 '[2/120]'
    # [4/120] caltech-101 | sepfpl | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.4 1 30 exp2 '[4/120]'
    # [6/120] caltech-101 | sepfpl_hcse | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 1 0.1 1 30 exp2 '[6/120]'
    # [8/120] caltech-101 | sepfpl | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.1 1 30 exp2 '[8/120]'
    # [10/120] caltech-101 | sepfpl_hcse | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 1 0.01 1 30 exp2 '[10/120]'
    # [12/120] caltech-101 | sepfpl | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.01 1 30 exp2 '[12/120]'
    # [14/120] caltech-101 | sepfpl_hcse | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 2 0.4 1 30 exp2 '[14/120]'
    # [16/120] caltech-101 | sepfpl | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.4 1 30 exp2 '[16/120]'
    # [18/120] caltech-101 | sepfpl_hcse | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 2 0.1 1 30 exp2 '[18/120]'
    # [20/120] caltech-101 | sepfpl | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.1 1 30 exp2 '[20/120]'
    # [22/120] caltech-101 | sepfpl_hcse | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 2 0.01 1 30 exp2 '[22/120]'
    # [24/120] caltech-101 | sepfpl | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.01 1 30 exp2 '[24/120]'
    # [26/120] caltech-101 | sepfpl_hcse | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 4 0.4 1 30 exp2 '[26/120]'
    # [28/120] caltech-101 | sepfpl | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.4 1 30 exp2 '[28/120]'
    # [30/120] caltech-101 | sepfpl_hcse | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 4 0.1 1 30 exp2 '[30/120]'
    # [32/120] caltech-101 | sepfpl | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.1 1 30 exp2 '[32/120]'
    # [34/120] caltech-101 | sepfpl_hcse | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 4 0.01 1 30 exp2 '[34/120]'
    # [36/120] caltech-101 | sepfpl | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.01 1 30 exp2 '[36/120]'
    # [38/120] caltech-101 | sepfpl_hcse | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.4 1 30 exp2 '[38/120]'
    # [40/120] caltech-101 | sepfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 30 exp2 '[40/120]'
    # [42/120] caltech-101 | sepfpl_hcse | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.1 1 30 exp2 '[42/120]'
    # [44/120] caltech-101 | sepfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 30 exp2 '[44/120]'
    # [46/120] caltech-101 | sepfpl_hcse | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.01 1 30 exp2 '[46/120]'
    # [48/120] caltech-101 | sepfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 30 exp2 '[48/120]'
    # [50/120] caltech-101 | sepfpl_hcse | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 16 0.4 1 30 exp2 '[50/120]'
    # [52/120] caltech-101 | sepfpl | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.4 1 30 exp2 '[52/120]'
    # [54/120] caltech-101 | sepfpl_hcse | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 16 0.1 1 30 exp2 '[54/120]'
    # [56/120] caltech-101 | sepfpl | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.1 1 30 exp2 '[56/120]'
    # [58/120] caltech-101 | sepfpl_hcse | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 16 0.01 1 30 exp2 '[58/120]'
    # [60/120] caltech-101 | sepfpl | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.01 1 30 exp2 '[60/120]'
    # [62/120] oxford_pets | sepfpl_hcse | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 1 0.4 1 30 exp2 '[62/120]'
    # [64/120] oxford_pets | sepfpl | r=1 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0.4 1 30 exp2 '[64/120]'
    # [66/120] oxford_pets | sepfpl_hcse | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 1 0.1 1 30 exp2 '[66/120]'
    # [68/120] oxford_pets | sepfpl | r=1 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0.1 1 30 exp2 '[68/120]'
    # [70/120] oxford_pets | sepfpl_hcse | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 1 0.01 1 30 exp2 '[70/120]'
    # [72/120] oxford_pets | sepfpl | r=1 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0.01 1 30 exp2 '[72/120]'
    # [74/120] oxford_pets | sepfpl_hcse | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 2 0.4 1 30 exp2 '[74/120]'
    # [76/120] oxford_pets | sepfpl | r=2 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0.4 1 30 exp2 '[76/120]'
    # [78/120] oxford_pets | sepfpl_hcse | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 2 0.1 1 30 exp2 '[78/120]'
    # [80/120] oxford_pets | sepfpl | r=2 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0.1 1 30 exp2 '[80/120]'
    # [82/120] oxford_pets | sepfpl_hcse | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 2 0.01 1 30 exp2 '[82/120]'
    # [84/120] oxford_pets | sepfpl | r=2 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0.01 1 30 exp2 '[84/120]'
    # [86/120] oxford_pets | sepfpl_hcse | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 4 0.4 1 30 exp2 '[86/120]'
    # [88/120] oxford_pets | sepfpl | r=4 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.4 1 30 exp2 '[88/120]'
    # [90/120] oxford_pets | sepfpl_hcse | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 4 0.1 1 30 exp2 '[90/120]'
    # [92/120] oxford_pets | sepfpl | r=4 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.1 1 30 exp2 '[92/120]'
    # [94/120] oxford_pets | sepfpl_hcse | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 4 0.01 1 30 exp2 '[94/120]'
    # [96/120] oxford_pets | sepfpl | r=4 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.01 1 30 exp2 '[96/120]'
    # [98/120] oxford_pets | sepfpl_hcse | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.4 1 30 exp2 '[98/120]'
    # [100/120] oxford_pets | sepfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 30 exp2 '[100/120]'
    # [102/120] oxford_pets | sepfpl_hcse | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.1 1 30 exp2 '[102/120]'
    # [104/120] oxford_pets | sepfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 30 exp2 '[104/120]'
    # [106/120] oxford_pets | sepfpl_hcse | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.01 1 30 exp2 '[106/120]'
    # [108/120] oxford_pets | sepfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 30 exp2 '[108/120]'
    # [110/120] oxford_pets | sepfpl_hcse | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 16 0.4 1 30 exp2 '[110/120]'
    # [112/120] oxford_pets | sepfpl | r=16 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.4 1 30 exp2 '[112/120]'
    # [114/120] oxford_pets | sepfpl_hcse | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 16 0.1 1 30 exp2 '[114/120]'
    # [116/120] oxford_pets | sepfpl | r=16 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.1 1 30 exp2 '[116/120]'
    # [118/120] oxford_pets | sepfpl_hcse | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 16 0.01 1 30 exp2 '[118/120]'
    # [120/120] oxford_pets | sepfpl | r=16 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.01 1 30 exp2 '[120/120]'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
