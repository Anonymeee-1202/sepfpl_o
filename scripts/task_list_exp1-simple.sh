#!/bin/bash

# 实验任务列表: exp1-simple
# 生成时间: 2025-11-20 23:17:04
# 任务总数: 120
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/120] caltech-101 | promptfl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.0 1 30 exp1-simple '[1/120]'
    # [3/120] caltech-101 | fedpgp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.0 1 30 exp1-simple '[3/120]'
    # [5/120] caltech-101 | sepfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 30 exp1-simple '[5/120]'
    # [7/120] caltech-101 | fedotp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.4 1 30 exp1-simple '[7/120]'
    # [9/120] caltech-101 | dpfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.4 1 30 exp1-simple '[9/120]'
    # [11/120] caltech-101 | promptfl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.2 1 30 exp1-simple '[11/120]'
    # [13/120] caltech-101 | fedpgp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.2 1 30 exp1-simple '[13/120]'
    # [15/120] caltech-101 | sepfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.2 1 30 exp1-simple '[15/120]'
    # [17/120] caltech-101 | fedotp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.1 1 30 exp1-simple '[17/120]'
    # [19/120] caltech-101 | dpfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.1 1 30 exp1-simple '[19/120]'
    # [21/120] caltech-101 | promptfl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.05 1 30 exp1-simple '[21/120]'
    # [23/120] caltech-101 | fedpgp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.05 1 30 exp1-simple '[23/120]'
    # [25/120] caltech-101 | sepfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.05 1 30 exp1-simple '[25/120]'
    # [27/120] caltech-101 | fedotp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.01 1 30 exp1-simple '[27/120]'
    # [29/120] caltech-101 | dpfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.01 1 30 exp1-simple '[29/120]'
    # [31/120] oxford_pets | promptfl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.0 1 30 exp1-simple '[31/120]'
    # [33/120] oxford_pets | fedpgp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.0 1 30 exp1-simple '[33/120]'
    # [35/120] oxford_pets | sepfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 1 30 exp1-simple '[35/120]'
    # [37/120] oxford_pets | fedotp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.4 1 30 exp1-simple '[37/120]'
    # [39/120] oxford_pets | dpfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.4 1 30 exp1-simple '[39/120]'
    # [41/120] oxford_pets | promptfl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.2 1 30 exp1-simple '[41/120]'
    # [43/120] oxford_pets | fedpgp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.2 1 30 exp1-simple '[43/120]'
    # [45/120] oxford_pets | sepfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.2 1 30 exp1-simple '[45/120]'
    # [47/120] oxford_pets | fedotp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.1 1 30 exp1-simple '[47/120]'
    # [49/120] oxford_pets | dpfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.1 1 30 exp1-simple '[49/120]'
    # [51/120] oxford_pets | promptfl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.05 1 30 exp1-simple '[51/120]'
    # [53/120] oxford_pets | fedpgp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.05 1 30 exp1-simple '[53/120]'
    # [55/120] oxford_pets | sepfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.05 1 30 exp1-simple '[55/120]'
    # [57/120] oxford_pets | fedotp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.01 1 30 exp1-simple '[57/120]'
    # [59/120] oxford_pets | dpfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.01 1 30 exp1-simple '[59/120]'
    # [61/120] oxford_flowers | promptfl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.0 1 30 exp1-simple '[61/120]'
    # [63/120] oxford_flowers | fedpgp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.0 1 30 exp1-simple '[63/120]'
    # [65/120] oxford_flowers | sepfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.0 1 30 exp1-simple '[65/120]'
    # [67/120] oxford_flowers | fedotp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.4 1 30 exp1-simple '[67/120]'
    # [69/120] oxford_flowers | dpfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.4 1 30 exp1-simple '[69/120]'
    # [71/120] oxford_flowers | promptfl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.2 1 30 exp1-simple '[71/120]'
    # [73/120] oxford_flowers | fedpgp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.2 1 30 exp1-simple '[73/120]'
    # [75/120] oxford_flowers | sepfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.2 1 30 exp1-simple '[75/120]'
    # [77/120] oxford_flowers | fedotp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.1 1 30 exp1-simple '[77/120]'
    # [79/120] oxford_flowers | dpfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.1 1 30 exp1-simple '[79/120]'
    # [81/120] oxford_flowers | promptfl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.05 1 30 exp1-simple '[81/120]'
    # [83/120] oxford_flowers | fedpgp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.05 1 30 exp1-simple '[83/120]'
    # [85/120] oxford_flowers | sepfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.05 1 30 exp1-simple '[85/120]'
    # [87/120] oxford_flowers | fedotp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.01 1 30 exp1-simple '[87/120]'
    # [89/120] oxford_flowers | dpfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.01 1 30 exp1-simple '[89/120]'
    # [91/120] food-101 | promptfl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.0 1 30 exp1-simple '[91/120]'
    # [93/120] food-101 | fedpgp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.0 1 30 exp1-simple '[93/120]'
    # [95/120] food-101 | sepfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.0 1 30 exp1-simple '[95/120]'
    # [97/120] food-101 | fedotp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.4 1 30 exp1-simple '[97/120]'
    # [99/120] food-101 | dpfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.4 1 30 exp1-simple '[99/120]'
    # [101/120] food-101 | promptfl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.2 1 30 exp1-simple '[101/120]'
    # [103/120] food-101 | fedpgp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.2 1 30 exp1-simple '[103/120]'
    # [105/120] food-101 | sepfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.2 1 30 exp1-simple '[105/120]'
    # [107/120] food-101 | fedotp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.1 1 30 exp1-simple '[107/120]'
    # [109/120] food-101 | dpfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.1 1 30 exp1-simple '[109/120]'
    # [111/120] food-101 | promptfl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.05 1 30 exp1-simple '[111/120]'
    # [113/120] food-101 | fedpgp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.05 1 30 exp1-simple '[113/120]'
    # [115/120] food-101 | sepfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.05 1 30 exp1-simple '[115/120]'
    # [117/120] food-101 | fedotp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.01 1 30 exp1-simple '[117/120]'
    # [119/120] food-101 | dpfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.01 1 30 exp1-simple '[119/120]'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/120] caltech-101 | fedotp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.0 1 30 exp1-simple '[2/120]'
    # [4/120] caltech-101 | dpfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.0 1 30 exp1-simple '[4/120]'
    # [6/120] caltech-101 | promptfl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.4 1 30 exp1-simple '[6/120]'
    # [8/120] caltech-101 | fedpgp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.4 1 30 exp1-simple '[8/120]'
    # [10/120] caltech-101 | sepfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 30 exp1-simple '[10/120]'
    # [12/120] caltech-101 | fedotp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.2 1 30 exp1-simple '[12/120]'
    # [14/120] caltech-101 | dpfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.2 1 30 exp1-simple '[14/120]'
    # [16/120] caltech-101 | promptfl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.1 1 30 exp1-simple '[16/120]'
    # [18/120] caltech-101 | fedpgp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.1 1 30 exp1-simple '[18/120]'
    # [20/120] caltech-101 | sepfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 30 exp1-simple '[20/120]'
    # [22/120] caltech-101 | fedotp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.05 1 30 exp1-simple '[22/120]'
    # [24/120] caltech-101 | dpfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.05 1 30 exp1-simple '[24/120]'
    # [26/120] caltech-101 | promptfl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.01 1 30 exp1-simple '[26/120]'
    # [28/120] caltech-101 | fedpgp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.01 1 30 exp1-simple '[28/120]'
    # [30/120] caltech-101 | sepfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 30 exp1-simple '[30/120]'
    # [32/120] oxford_pets | fedotp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.0 1 30 exp1-simple '[32/120]'
    # [34/120] oxford_pets | dpfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.0 1 30 exp1-simple '[34/120]'
    # [36/120] oxford_pets | promptfl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.4 1 30 exp1-simple '[36/120]'
    # [38/120] oxford_pets | fedpgp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.4 1 30 exp1-simple '[38/120]'
    # [40/120] oxford_pets | sepfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 30 exp1-simple '[40/120]'
    # [42/120] oxford_pets | fedotp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.2 1 30 exp1-simple '[42/120]'
    # [44/120] oxford_pets | dpfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.2 1 30 exp1-simple '[44/120]'
    # [46/120] oxford_pets | promptfl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.1 1 30 exp1-simple '[46/120]'
    # [48/120] oxford_pets | fedpgp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.1 1 30 exp1-simple '[48/120]'
    # [50/120] oxford_pets | sepfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 30 exp1-simple '[50/120]'
    # [52/120] oxford_pets | fedotp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.05 1 30 exp1-simple '[52/120]'
    # [54/120] oxford_pets | dpfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.05 1 30 exp1-simple '[54/120]'
    # [56/120] oxford_pets | promptfl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.01 1 30 exp1-simple '[56/120]'
    # [58/120] oxford_pets | fedpgp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.01 1 30 exp1-simple '[58/120]'
    # [60/120] oxford_pets | sepfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 30 exp1-simple '[60/120]'
    # [62/120] oxford_flowers | fedotp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.0 1 30 exp1-simple '[62/120]'
    # [64/120] oxford_flowers | dpfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.0 1 30 exp1-simple '[64/120]'
    # [66/120] oxford_flowers | promptfl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.4 1 30 exp1-simple '[66/120]'
    # [68/120] oxford_flowers | fedpgp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.4 1 30 exp1-simple '[68/120]'
    # [70/120] oxford_flowers | sepfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.4 1 30 exp1-simple '[70/120]'
    # [72/120] oxford_flowers | fedotp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.2 1 30 exp1-simple '[72/120]'
    # [74/120] oxford_flowers | dpfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.2 1 30 exp1-simple '[74/120]'
    # [76/120] oxford_flowers | promptfl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.1 1 30 exp1-simple '[76/120]'
    # [78/120] oxford_flowers | fedpgp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.1 1 30 exp1-simple '[78/120]'
    # [80/120] oxford_flowers | sepfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.1 1 30 exp1-simple '[80/120]'
    # [82/120] oxford_flowers | fedotp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.05 1 30 exp1-simple '[82/120]'
    # [84/120] oxford_flowers | dpfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.05 1 30 exp1-simple '[84/120]'
    # [86/120] oxford_flowers | promptfl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.01 1 30 exp1-simple '[86/120]'
    # [88/120] oxford_flowers | fedpgp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.01 1 30 exp1-simple '[88/120]'
    # [90/120] oxford_flowers | sepfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.01 1 30 exp1-simple '[90/120]'
    # [92/120] food-101 | fedotp | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.0 1 30 exp1-simple '[92/120]'
    # [94/120] food-101 | dpfpl | r=8 n=0.0 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.0 1 30 exp1-simple '[94/120]'
    # [96/120] food-101 | promptfl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.4 1 30 exp1-simple '[96/120]'
    # [98/120] food-101 | fedpgp | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.4 1 30 exp1-simple '[98/120]'
    # [100/120] food-101 | sepfpl | r=8 n=0.4 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.4 1 30 exp1-simple '[100/120]'
    # [102/120] food-101 | fedotp | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.2 1 30 exp1-simple '[102/120]'
    # [104/120] food-101 | dpfpl | r=8 n=0.2 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.2 1 30 exp1-simple '[104/120]'
    # [106/120] food-101 | promptfl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.1 1 30 exp1-simple '[106/120]'
    # [108/120] food-101 | fedpgp | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.1 1 30 exp1-simple '[108/120]'
    # [110/120] food-101 | sepfpl | r=8 n=0.1 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.1 1 30 exp1-simple '[110/120]'
    # [112/120] food-101 | fedotp | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.05 1 30 exp1-simple '[112/120]'
    # [114/120] food-101 | dpfpl | r=8 n=0.05 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.05 1 30 exp1-simple '[114/120]'
    # [116/120] food-101 | promptfl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.01 1 30 exp1-simple '[116/120]'
    # [118/120] food-101 | fedpgp | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.01 1 30 exp1-simple '[118/120]'
    # [120/120] food-101 | sepfpl | r=8 n=0.01 u=10 s=1
    CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.01 1 30 exp1-simple '[120/120]'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
