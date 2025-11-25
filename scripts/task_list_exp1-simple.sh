#!/bin/bash

# 实验任务列表: exp1-simple
# 生成时间: 2025-11-25 15:13:39
# 任务总数: 120
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 顺序执行模式 (无 GPU 或单 GPU)
echo '▶️  正在执行任务 [1/120]: caltech-101 | promptfl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.0 1 30 exp1-simple '[1/120]'

echo '▶️  正在执行任务 [2/120]: caltech-101 | fedotp | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.0 1 30 exp1-simple '[2/120]'

echo '▶️  正在执行任务 [3/120]: caltech-101 | fedpgp | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.0 1 30 exp1-simple '[3/120]'

echo '▶️  正在执行任务 [4/120]: caltech-101 | dpfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.0 1 30 exp1-simple '[4/120]'

echo '▶️  正在执行任务 [5/120]: caltech-101 | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 30 exp1-simple '[5/120]'

echo '▶️  正在执行任务 [6/120]: caltech-101 | promptfl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.4 1 30 exp1-simple '[6/120]'

echo '▶️  正在执行任务 [7/120]: caltech-101 | fedotp | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.4 1 30 exp1-simple '[7/120]'

echo '▶️  正在执行任务 [8/120]: caltech-101 | fedpgp | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.4 1 30 exp1-simple '[8/120]'

echo '▶️  正在执行任务 [9/120]: caltech-101 | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.4 1 30 exp1-simple '[9/120]'

echo '▶️  正在执行任务 [10/120]: caltech-101 | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 30 exp1-simple '[10/120]'

echo '▶️  正在执行任务 [11/120]: caltech-101 | promptfl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.2 1 30 exp1-simple '[11/120]'

echo '▶️  正在执行任务 [12/120]: caltech-101 | fedotp | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.2 1 30 exp1-simple '[12/120]'

echo '▶️  正在执行任务 [13/120]: caltech-101 | fedpgp | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.2 1 30 exp1-simple '[13/120]'

echo '▶️  正在执行任务 [14/120]: caltech-101 | dpfpl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.2 1 30 exp1-simple '[14/120]'

echo '▶️  正在执行任务 [15/120]: caltech-101 | sepfpl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.2 1 30 exp1-simple '[15/120]'

echo '▶️  正在执行任务 [16/120]: caltech-101 | promptfl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.1 1 30 exp1-simple '[16/120]'

echo '▶️  正在执行任务 [17/120]: caltech-101 | fedotp | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.1 1 30 exp1-simple '[17/120]'

echo '▶️  正在执行任务 [18/120]: caltech-101 | fedpgp | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.1 1 30 exp1-simple '[18/120]'

echo '▶️  正在执行任务 [19/120]: caltech-101 | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.1 1 30 exp1-simple '[19/120]'

echo '▶️  正在执行任务 [20/120]: caltech-101 | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 30 exp1-simple '[20/120]'

echo '▶️  正在执行任务 [21/120]: caltech-101 | promptfl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.05 1 30 exp1-simple '[21/120]'

echo '▶️  正在执行任务 [22/120]: caltech-101 | fedotp | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.05 1 30 exp1-simple '[22/120]'

echo '▶️  正在执行任务 [23/120]: caltech-101 | fedpgp | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.05 1 30 exp1-simple '[23/120]'

echo '▶️  正在执行任务 [24/120]: caltech-101 | dpfpl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.05 1 30 exp1-simple '[24/120]'

echo '▶️  正在执行任务 [25/120]: caltech-101 | sepfpl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.05 1 30 exp1-simple '[25/120]'

echo '▶️  正在执行任务 [26/120]: caltech-101 | promptfl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.01 1 30 exp1-simple '[26/120]'

echo '▶️  正在执行任务 [27/120]: caltech-101 | fedotp | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.01 1 30 exp1-simple '[27/120]'

echo '▶️  正在执行任务 [28/120]: caltech-101 | fedpgp | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.01 1 30 exp1-simple '[28/120]'

echo '▶️  正在执行任务 [29/120]: caltech-101 | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.01 1 30 exp1-simple '[29/120]'

echo '▶️  正在执行任务 [30/120]: caltech-101 | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 30 exp1-simple '[30/120]'

echo '▶️  正在执行任务 [31/120]: oxford_pets | promptfl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.0 1 30 exp1-simple '[31/120]'

echo '▶️  正在执行任务 [32/120]: oxford_pets | fedotp | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.0 1 30 exp1-simple '[32/120]'

echo '▶️  正在执行任务 [33/120]: oxford_pets | fedpgp | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.0 1 30 exp1-simple '[33/120]'

echo '▶️  正在执行任务 [34/120]: oxford_pets | dpfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.0 1 30 exp1-simple '[34/120]'

echo '▶️  正在执行任务 [35/120]: oxford_pets | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 1 30 exp1-simple '[35/120]'

echo '▶️  正在执行任务 [36/120]: oxford_pets | promptfl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.4 1 30 exp1-simple '[36/120]'

echo '▶️  正在执行任务 [37/120]: oxford_pets | fedotp | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.4 1 30 exp1-simple '[37/120]'

echo '▶️  正在执行任务 [38/120]: oxford_pets | fedpgp | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.4 1 30 exp1-simple '[38/120]'

echo '▶️  正在执行任务 [39/120]: oxford_pets | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.4 1 30 exp1-simple '[39/120]'

echo '▶️  正在执行任务 [40/120]: oxford_pets | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 30 exp1-simple '[40/120]'

echo '▶️  正在执行任务 [41/120]: oxford_pets | promptfl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.2 1 30 exp1-simple '[41/120]'

echo '▶️  正在执行任务 [42/120]: oxford_pets | fedotp | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.2 1 30 exp1-simple '[42/120]'

echo '▶️  正在执行任务 [43/120]: oxford_pets | fedpgp | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.2 1 30 exp1-simple '[43/120]'

echo '▶️  正在执行任务 [44/120]: oxford_pets | dpfpl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.2 1 30 exp1-simple '[44/120]'

echo '▶️  正在执行任务 [45/120]: oxford_pets | sepfpl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.2 1 30 exp1-simple '[45/120]'

echo '▶️  正在执行任务 [46/120]: oxford_pets | promptfl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.1 1 30 exp1-simple '[46/120]'

echo '▶️  正在执行任务 [47/120]: oxford_pets | fedotp | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.1 1 30 exp1-simple '[47/120]'

echo '▶️  正在执行任务 [48/120]: oxford_pets | fedpgp | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.1 1 30 exp1-simple '[48/120]'

echo '▶️  正在执行任务 [49/120]: oxford_pets | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.1 1 30 exp1-simple '[49/120]'

echo '▶️  正在执行任务 [50/120]: oxford_pets | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 30 exp1-simple '[50/120]'

echo '▶️  正在执行任务 [51/120]: oxford_pets | promptfl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.05 1 30 exp1-simple '[51/120]'

echo '▶️  正在执行任务 [52/120]: oxford_pets | fedotp | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.05 1 30 exp1-simple '[52/120]'

echo '▶️  正在执行任务 [53/120]: oxford_pets | fedpgp | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.05 1 30 exp1-simple '[53/120]'

echo '▶️  正在执行任务 [54/120]: oxford_pets | dpfpl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.05 1 30 exp1-simple '[54/120]'

echo '▶️  正在执行任务 [55/120]: oxford_pets | sepfpl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.05 1 30 exp1-simple '[55/120]'

echo '▶️  正在执行任务 [56/120]: oxford_pets | promptfl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.01 1 30 exp1-simple '[56/120]'

echo '▶️  正在执行任务 [57/120]: oxford_pets | fedotp | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.01 1 30 exp1-simple '[57/120]'

echo '▶️  正在执行任务 [58/120]: oxford_pets | fedpgp | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.01 1 30 exp1-simple '[58/120]'

echo '▶️  正在执行任务 [59/120]: oxford_pets | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.01 1 30 exp1-simple '[59/120]'

echo '▶️  正在执行任务 [60/120]: oxford_pets | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 30 exp1-simple '[60/120]'

echo '▶️  正在执行任务 [61/120]: oxford_flowers | promptfl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.0 1 30 exp1-simple '[61/120]'

echo '▶️  正在执行任务 [62/120]: oxford_flowers | fedotp | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.0 1 30 exp1-simple '[62/120]'

echo '▶️  正在执行任务 [63/120]: oxford_flowers | fedpgp | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.0 1 30 exp1-simple '[63/120]'

echo '▶️  正在执行任务 [64/120]: oxford_flowers | dpfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.0 1 30 exp1-simple '[64/120]'

echo '▶️  正在执行任务 [65/120]: oxford_flowers | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.0 1 30 exp1-simple '[65/120]'

echo '▶️  正在执行任务 [66/120]: oxford_flowers | promptfl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.4 1 30 exp1-simple '[66/120]'

echo '▶️  正在执行任务 [67/120]: oxford_flowers | fedotp | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.4 1 30 exp1-simple '[67/120]'

echo '▶️  正在执行任务 [68/120]: oxford_flowers | fedpgp | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.4 1 30 exp1-simple '[68/120]'

echo '▶️  正在执行任务 [69/120]: oxford_flowers | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.4 1 30 exp1-simple '[69/120]'

echo '▶️  正在执行任务 [70/120]: oxford_flowers | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.4 1 30 exp1-simple '[70/120]'

echo '▶️  正在执行任务 [71/120]: oxford_flowers | promptfl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.2 1 30 exp1-simple '[71/120]'

echo '▶️  正在执行任务 [72/120]: oxford_flowers | fedotp | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.2 1 30 exp1-simple '[72/120]'

echo '▶️  正在执行任务 [73/120]: oxford_flowers | fedpgp | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.2 1 30 exp1-simple '[73/120]'

echo '▶️  正在执行任务 [74/120]: oxford_flowers | dpfpl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.2 1 30 exp1-simple '[74/120]'

echo '▶️  正在执行任务 [75/120]: oxford_flowers | sepfpl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.2 1 30 exp1-simple '[75/120]'

echo '▶️  正在执行任务 [76/120]: oxford_flowers | promptfl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.1 1 30 exp1-simple '[76/120]'

echo '▶️  正在执行任务 [77/120]: oxford_flowers | fedotp | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.1 1 30 exp1-simple '[77/120]'

echo '▶️  正在执行任务 [78/120]: oxford_flowers | fedpgp | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.1 1 30 exp1-simple '[78/120]'

echo '▶️  正在执行任务 [79/120]: oxford_flowers | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.1 1 30 exp1-simple '[79/120]'

echo '▶️  正在执行任务 [80/120]: oxford_flowers | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.1 1 30 exp1-simple '[80/120]'

echo '▶️  正在执行任务 [81/120]: oxford_flowers | promptfl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.05 1 30 exp1-simple '[81/120]'

echo '▶️  正在执行任务 [82/120]: oxford_flowers | fedotp | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.05 1 30 exp1-simple '[82/120]'

echo '▶️  正在执行任务 [83/120]: oxford_flowers | fedpgp | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.05 1 30 exp1-simple '[83/120]'

echo '▶️  正在执行任务 [84/120]: oxford_flowers | dpfpl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.05 1 30 exp1-simple '[84/120]'

echo '▶️  正在执行任务 [85/120]: oxford_flowers | sepfpl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.05 1 30 exp1-simple '[85/120]'

echo '▶️  正在执行任务 [86/120]: oxford_flowers | promptfl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.01 1 30 exp1-simple '[86/120]'

echo '▶️  正在执行任务 [87/120]: oxford_flowers | fedotp | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.01 1 30 exp1-simple '[87/120]'

echo '▶️  正在执行任务 [88/120]: oxford_flowers | fedpgp | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.01 1 30 exp1-simple '[88/120]'

echo '▶️  正在执行任务 [89/120]: oxford_flowers | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.01 1 30 exp1-simple '[89/120]'

echo '▶️  正在执行任务 [90/120]: oxford_flowers | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.01 1 30 exp1-simple '[90/120]'

echo '▶️  正在执行任务 [91/120]: food-101 | promptfl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.0 1 30 exp1-simple '[91/120]'

echo '▶️  正在执行任务 [92/120]: food-101 | fedotp | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.0 1 30 exp1-simple '[92/120]'

echo '▶️  正在执行任务 [93/120]: food-101 | fedpgp | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.0 1 30 exp1-simple '[93/120]'

echo '▶️  正在执行任务 [94/120]: food-101 | dpfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.0 1 30 exp1-simple '[94/120]'

echo '▶️  正在执行任务 [95/120]: food-101 | sepfpl | r=8 n=0.0 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.0 1 30 exp1-simple '[95/120]'

echo '▶️  正在执行任务 [96/120]: food-101 | promptfl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.4 1 30 exp1-simple '[96/120]'

echo '▶️  正在执行任务 [97/120]: food-101 | fedotp | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.4 1 30 exp1-simple '[97/120]'

echo '▶️  正在执行任务 [98/120]: food-101 | fedpgp | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.4 1 30 exp1-simple '[98/120]'

echo '▶️  正在执行任务 [99/120]: food-101 | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.4 1 30 exp1-simple '[99/120]'

echo '▶️  正在执行任务 [100/120]: food-101 | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.4 1 30 exp1-simple '[100/120]'

echo '▶️  正在执行任务 [101/120]: food-101 | promptfl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.2 1 30 exp1-simple '[101/120]'

echo '▶️  正在执行任务 [102/120]: food-101 | fedotp | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.2 1 30 exp1-simple '[102/120]'

echo '▶️  正在执行任务 [103/120]: food-101 | fedpgp | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.2 1 30 exp1-simple '[103/120]'

echo '▶️  正在执行任务 [104/120]: food-101 | dpfpl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.2 1 30 exp1-simple '[104/120]'

echo '▶️  正在执行任务 [105/120]: food-101 | sepfpl | r=8 n=0.2 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.2 1 30 exp1-simple '[105/120]'

echo '▶️  正在执行任务 [106/120]: food-101 | promptfl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.1 1 30 exp1-simple '[106/120]'

echo '▶️  正在执行任务 [107/120]: food-101 | fedotp | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.1 1 30 exp1-simple '[107/120]'

echo '▶️  正在执行任务 [108/120]: food-101 | fedpgp | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.1 1 30 exp1-simple '[108/120]'

echo '▶️  正在执行任务 [109/120]: food-101 | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.1 1 30 exp1-simple '[109/120]'

echo '▶️  正在执行任务 [110/120]: food-101 | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.1 1 30 exp1-simple '[110/120]'

echo '▶️  正在执行任务 [111/120]: food-101 | promptfl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.05 1 30 exp1-simple '[111/120]'

echo '▶️  正在执行任务 [112/120]: food-101 | fedotp | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.05 1 30 exp1-simple '[112/120]'

echo '▶️  正在执行任务 [113/120]: food-101 | fedpgp | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.05 1 30 exp1-simple '[113/120]'

echo '▶️  正在执行任务 [114/120]: food-101 | dpfpl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.05 1 30 exp1-simple '[114/120]'

echo '▶️  正在执行任务 [115/120]: food-101 | sepfpl | r=8 n=0.05 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.05 1 30 exp1-simple '[115/120]'

echo '▶️  正在执行任务 [116/120]: food-101 | promptfl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.01 1 30 exp1-simple '[116/120]'

echo '▶️  正在执行任务 [117/120]: food-101 | fedotp | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.01 1 30 exp1-simple '[117/120]'

echo '▶️  正在执行任务 [118/120]: food-101 | fedpgp | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.01 1 30 exp1-simple '[118/120]'

echo '▶️  正在执行任务 [119/120]: food-101 | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.01 1 30 exp1-simple '[119/120]'

echo '▶️  正在执行任务 [120/120]: food-101 | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.01 1 30 exp1-simple '[120/120]'

