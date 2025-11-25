#!/bin/bash

# 实验任务列表: exp1-hard
# 生成时间: 2025-11-25 16:17:16
# 任务总数: 60
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 顺序执行模式 (无 GPU 或单 GPU)
echo '▶️  正在执行任务 [1/60]: cifar-100 | promptfl | r=8 n=0.0 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.0 1 30 exp1-hard '[1/60]'

echo '▶️  正在执行任务 [2/60]: cifar-100 | fedotp | r=8 n=0.0 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.0 1 30 exp1-hard '[2/60]'

echo '▶️  正在执行任务 [3/60]: cifar-100 | fedpgp | r=8 n=0.0 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.0 1 30 exp1-hard '[3/60]'

echo '▶️  正在执行任务 [4/60]: cifar-100 | dpfpl | r=8 n=0.0 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.0 1 30 exp1-hard '[4/60]'

echo '▶️  正在执行任务 [5/60]: cifar-100 | sepfpl | r=8 n=0.0 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.0 1 30 exp1-hard '[5/60]'

echo '▶️  正在执行任务 [6/60]: cifar-100 | promptfl | r=8 n=0.4 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.4 1 30 exp1-hard '[6/60]'

echo '▶️  正在执行任务 [7/60]: cifar-100 | fedotp | r=8 n=0.4 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.4 1 30 exp1-hard '[7/60]'

echo '▶️  正在执行任务 [8/60]: cifar-100 | fedpgp | r=8 n=0.4 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.4 1 30 exp1-hard '[8/60]'

echo '▶️  正在执行任务 [9/60]: cifar-100 | dpfpl | r=8 n=0.4 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.4 1 30 exp1-hard '[9/60]'

echo '▶️  正在执行任务 [10/60]: cifar-100 | sepfpl | r=8 n=0.4 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.4 1 30 exp1-hard '[10/60]'

echo '▶️  正在执行任务 [11/60]: cifar-100 | promptfl | r=8 n=0.2 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.2 1 30 exp1-hard '[11/60]'

echo '▶️  正在执行任务 [12/60]: cifar-100 | fedotp | r=8 n=0.2 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.2 1 30 exp1-hard '[12/60]'

echo '▶️  正在执行任务 [13/60]: cifar-100 | fedpgp | r=8 n=0.2 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.2 1 30 exp1-hard '[13/60]'

echo '▶️  正在执行任务 [14/60]: cifar-100 | dpfpl | r=8 n=0.2 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.2 1 30 exp1-hard '[14/60]'

echo '▶️  正在执行任务 [15/60]: cifar-100 | sepfpl | r=8 n=0.2 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.2 1 30 exp1-hard '[15/60]'

echo '▶️  正在执行任务 [16/60]: cifar-100 | promptfl | r=8 n=0.1 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.1 1 30 exp1-hard '[16/60]'

echo '▶️  正在执行任务 [17/60]: cifar-100 | fedotp | r=8 n=0.1 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.1 1 30 exp1-hard '[17/60]'

echo '▶️  正在执行任务 [18/60]: cifar-100 | fedpgp | r=8 n=0.1 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.1 1 30 exp1-hard '[18/60]'

echo '▶️  正在执行任务 [19/60]: cifar-100 | dpfpl | r=8 n=0.1 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.1 1 30 exp1-hard '[19/60]'

echo '▶️  正在执行任务 [20/60]: cifar-100 | sepfpl | r=8 n=0.1 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.1 1 30 exp1-hard '[20/60]'

echo '▶️  正在执行任务 [21/60]: cifar-100 | promptfl | r=8 n=0.05 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.05 1 30 exp1-hard '[21/60]'

echo '▶️  正在执行任务 [22/60]: cifar-100 | fedotp | r=8 n=0.05 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.05 1 30 exp1-hard '[22/60]'

echo '▶️  正在执行任务 [23/60]: cifar-100 | fedpgp | r=8 n=0.05 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.05 1 30 exp1-hard '[23/60]'

echo '▶️  正在执行任务 [24/60]: cifar-100 | dpfpl | r=8 n=0.05 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.05 1 30 exp1-hard '[24/60]'

echo '▶️  正在执行任务 [25/60]: cifar-100 | sepfpl | r=8 n=0.05 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.05 1 30 exp1-hard '[25/60]'

echo '▶️  正在执行任务 [26/60]: cifar-100 | promptfl | r=8 n=0.01 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.01 1 30 exp1-hard '[26/60]'

echo '▶️  正在执行任务 [27/60]: cifar-100 | fedotp | r=8 n=0.01 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.01 1 30 exp1-hard '[27/60]'

echo '▶️  正在执行任务 [28/60]: cifar-100 | fedpgp | r=8 n=0.01 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.01 1 30 exp1-hard '[28/60]'

echo '▶️  正在执行任务 [29/60]: cifar-100 | dpfpl | r=8 n=0.01 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.01 1 30 exp1-hard '[29/60]'

echo '▶️  正在执行任务 [30/60]: cifar-100 | sepfpl | r=8 n=0.01 u=25 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.01 1 30 exp1-hard '[30/60]'

echo '▶️  正在执行任务 [31/60]: cifar-100 | promptfl | r=8 n=0.0 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.0 1 30 exp1-hard '[31/60]'

echo '▶️  正在执行任务 [32/60]: cifar-100 | fedotp | r=8 n=0.0 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.0 1 30 exp1-hard '[32/60]'

echo '▶️  正在执行任务 [33/60]: cifar-100 | fedpgp | r=8 n=0.0 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.0 1 30 exp1-hard '[33/60]'

echo '▶️  正在执行任务 [34/60]: cifar-100 | dpfpl | r=8 n=0.0 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.0 1 30 exp1-hard '[34/60]'

echo '▶️  正在执行任务 [35/60]: cifar-100 | sepfpl | r=8 n=0.0 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.0 1 30 exp1-hard '[35/60]'

echo '▶️  正在执行任务 [36/60]: cifar-100 | promptfl | r=8 n=0.4 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.4 1 30 exp1-hard '[36/60]'

echo '▶️  正在执行任务 [37/60]: cifar-100 | fedotp | r=8 n=0.4 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.4 1 30 exp1-hard '[37/60]'

echo '▶️  正在执行任务 [38/60]: cifar-100 | fedpgp | r=8 n=0.4 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.4 1 30 exp1-hard '[38/60]'

echo '▶️  正在执行任务 [39/60]: cifar-100 | dpfpl | r=8 n=0.4 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.4 1 30 exp1-hard '[39/60]'

echo '▶️  正在执行任务 [40/60]: cifar-100 | sepfpl | r=8 n=0.4 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.4 1 30 exp1-hard '[40/60]'

echo '▶️  正在执行任务 [41/60]: cifar-100 | promptfl | r=8 n=0.2 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.2 1 30 exp1-hard '[41/60]'

echo '▶️  正在执行任务 [42/60]: cifar-100 | fedotp | r=8 n=0.2 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.2 1 30 exp1-hard '[42/60]'

echo '▶️  正在执行任务 [43/60]: cifar-100 | fedpgp | r=8 n=0.2 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.2 1 30 exp1-hard '[43/60]'

echo '▶️  正在执行任务 [44/60]: cifar-100 | dpfpl | r=8 n=0.2 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.2 1 30 exp1-hard '[44/60]'

echo '▶️  正在执行任务 [45/60]: cifar-100 | sepfpl | r=8 n=0.2 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.2 1 30 exp1-hard '[45/60]'

echo '▶️  正在执行任务 [46/60]: cifar-100 | promptfl | r=8 n=0.1 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.1 1 30 exp1-hard '[46/60]'

echo '▶️  正在执行任务 [47/60]: cifar-100 | fedotp | r=8 n=0.1 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.1 1 30 exp1-hard '[47/60]'

echo '▶️  正在执行任务 [48/60]: cifar-100 | fedpgp | r=8 n=0.1 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.1 1 30 exp1-hard '[48/60]'

echo '▶️  正在执行任务 [49/60]: cifar-100 | dpfpl | r=8 n=0.1 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.1 1 30 exp1-hard '[49/60]'

echo '▶️  正在执行任务 [50/60]: cifar-100 | sepfpl | r=8 n=0.1 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.1 1 30 exp1-hard '[50/60]'

echo '▶️  正在执行任务 [51/60]: cifar-100 | promptfl | r=8 n=0.05 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.05 1 30 exp1-hard '[51/60]'

echo '▶️  正在执行任务 [52/60]: cifar-100 | fedotp | r=8 n=0.05 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.05 1 30 exp1-hard '[52/60]'

echo '▶️  正在执行任务 [53/60]: cifar-100 | fedpgp | r=8 n=0.05 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.05 1 30 exp1-hard '[53/60]'

echo '▶️  正在执行任务 [54/60]: cifar-100 | dpfpl | r=8 n=0.05 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.05 1 30 exp1-hard '[54/60]'

echo '▶️  正在执行任务 [55/60]: cifar-100 | sepfpl | r=8 n=0.05 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.05 1 30 exp1-hard '[55/60]'

echo '▶️  正在执行任务 [56/60]: cifar-100 | promptfl | r=8 n=0.01 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.01 1 30 exp1-hard '[56/60]'

echo '▶️  正在执行任务 [57/60]: cifar-100 | fedotp | r=8 n=0.01 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.01 1 30 exp1-hard '[57/60]'

echo '▶️  正在执行任务 [58/60]: cifar-100 | fedpgp | r=8 n=0.01 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.01 1 30 exp1-hard '[58/60]'

echo '▶️  正在执行任务 [59/60]: cifar-100 | dpfpl | r=8 n=0.01 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.01 1 30 exp1-hard '[59/60]'

echo '▶️  正在执行任务 [60/60]: cifar-100 | sepfpl | r=8 n=0.01 u=50 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.01 1 30 exp1-hard '[60/60]'

