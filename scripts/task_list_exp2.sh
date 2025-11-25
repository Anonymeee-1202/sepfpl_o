#!/bin/bash

# 实验任务列表: exp2
# 生成时间: 2025-11-25 14:39:16
# 任务总数: 120
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 顺序执行模式 (无 GPU 或单 GPU)
echo '▶️  正在执行任务 [1/120]: caltech-101 | dpfpl | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 1 0.4 1 40 exp2 '[1/120]'

echo '▶️  正在执行任务 [2/120]: caltech-101 | sepfpl_hcse | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 1 0.4 1 40 exp2 '[2/120]'

echo '▶️  正在执行任务 [3/120]: caltech-101 | sepfpl_time_adaptive | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.4 1 40 exp2 '[3/120]'

echo '▶️  正在执行任务 [4/120]: caltech-101 | sepfpl | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.4 1 40 exp2 '[4/120]'

echo '▶️  正在执行任务 [5/120]: caltech-101 | dpfpl | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 1 0.1 1 40 exp2 '[5/120]'

echo '▶️  正在执行任务 [6/120]: caltech-101 | sepfpl_hcse | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 1 0.1 1 40 exp2 '[6/120]'

echo '▶️  正在执行任务 [7/120]: caltech-101 | sepfpl_time_adaptive | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.1 1 40 exp2 '[7/120]'

echo '▶️  正在执行任务 [8/120]: caltech-101 | sepfpl | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.1 1 40 exp2 '[8/120]'

echo '▶️  正在执行任务 [9/120]: caltech-101 | dpfpl | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 1 0.01 1 40 exp2 '[9/120]'

echo '▶️  正在执行任务 [10/120]: caltech-101 | sepfpl_hcse | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 1 0.01 1 40 exp2 '[10/120]'

echo '▶️  正在执行任务 [11/120]: caltech-101 | sepfpl_time_adaptive | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.01 1 40 exp2 '[11/120]'

echo '▶️  正在执行任务 [12/120]: caltech-101 | sepfpl | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.01 1 40 exp2 '[12/120]'

echo '▶️  正在执行任务 [13/120]: caltech-101 | dpfpl | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 2 0.4 1 40 exp2 '[13/120]'

echo '▶️  正在执行任务 [14/120]: caltech-101 | sepfpl_hcse | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 2 0.4 1 40 exp2 '[14/120]'

echo '▶️  正在执行任务 [15/120]: caltech-101 | sepfpl_time_adaptive | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.4 1 40 exp2 '[15/120]'

echo '▶️  正在执行任务 [16/120]: caltech-101 | sepfpl | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.4 1 40 exp2 '[16/120]'

echo '▶️  正在执行任务 [17/120]: caltech-101 | dpfpl | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 2 0.1 1 40 exp2 '[17/120]'

echo '▶️  正在执行任务 [18/120]: caltech-101 | sepfpl_hcse | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 2 0.1 1 40 exp2 '[18/120]'

echo '▶️  正在执行任务 [19/120]: caltech-101 | sepfpl_time_adaptive | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.1 1 40 exp2 '[19/120]'

echo '▶️  正在执行任务 [20/120]: caltech-101 | sepfpl | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.1 1 40 exp2 '[20/120]'

echo '▶️  正在执行任务 [21/120]: caltech-101 | dpfpl | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 2 0.01 1 40 exp2 '[21/120]'

echo '▶️  正在执行任务 [22/120]: caltech-101 | sepfpl_hcse | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 2 0.01 1 40 exp2 '[22/120]'

echo '▶️  正在执行任务 [23/120]: caltech-101 | sepfpl_time_adaptive | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.01 1 40 exp2 '[23/120]'

echo '▶️  正在执行任务 [24/120]: caltech-101 | sepfpl | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.01 1 40 exp2 '[24/120]'

echo '▶️  正在执行任务 [25/120]: caltech-101 | dpfpl | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 4 0.4 1 40 exp2 '[25/120]'

echo '▶️  正在执行任务 [26/120]: caltech-101 | sepfpl_hcse | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 4 0.4 1 40 exp2 '[26/120]'

echo '▶️  正在执行任务 [27/120]: caltech-101 | sepfpl_time_adaptive | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.4 1 40 exp2 '[27/120]'

echo '▶️  正在执行任务 [28/120]: caltech-101 | sepfpl | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.4 1 40 exp2 '[28/120]'

echo '▶️  正在执行任务 [29/120]: caltech-101 | dpfpl | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 4 0.1 1 40 exp2 '[29/120]'

echo '▶️  正在执行任务 [30/120]: caltech-101 | sepfpl_hcse | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 4 0.1 1 40 exp2 '[30/120]'

echo '▶️  正在执行任务 [31/120]: caltech-101 | sepfpl_time_adaptive | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.1 1 40 exp2 '[31/120]'

echo '▶️  正在执行任务 [32/120]: caltech-101 | sepfpl | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.1 1 40 exp2 '[32/120]'

echo '▶️  正在执行任务 [33/120]: caltech-101 | dpfpl | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 4 0.01 1 40 exp2 '[33/120]'

echo '▶️  正在执行任务 [34/120]: caltech-101 | sepfpl_hcse | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 4 0.01 1 40 exp2 '[34/120]'

echo '▶️  正在执行任务 [35/120]: caltech-101 | sepfpl_time_adaptive | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.01 1 40 exp2 '[35/120]'

echo '▶️  正在执行任务 [36/120]: caltech-101 | sepfpl | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.01 1 40 exp2 '[36/120]'

echo '▶️  正在执行任务 [37/120]: caltech-101 | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.4 1 40 exp2 '[37/120]'

echo '▶️  正在执行任务 [38/120]: caltech-101 | sepfpl_hcse | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.4 1 40 exp2 '[38/120]'

echo '▶️  正在执行任务 [39/120]: caltech-101 | sepfpl_time_adaptive | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.4 1 40 exp2 '[39/120]'

echo '▶️  正在执行任务 [40/120]: caltech-101 | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 40 exp2 '[40/120]'

echo '▶️  正在执行任务 [41/120]: caltech-101 | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.1 1 40 exp2 '[41/120]'

echo '▶️  正在执行任务 [42/120]: caltech-101 | sepfpl_hcse | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.1 1 40 exp2 '[42/120]'

echo '▶️  正在执行任务 [43/120]: caltech-101 | sepfpl_time_adaptive | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.1 1 40 exp2 '[43/120]'

echo '▶️  正在执行任务 [44/120]: caltech-101 | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 40 exp2 '[44/120]'

echo '▶️  正在执行任务 [45/120]: caltech-101 | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.01 1 40 exp2 '[45/120]'

echo '▶️  正在执行任务 [46/120]: caltech-101 | sepfpl_hcse | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.01 1 40 exp2 '[46/120]'

echo '▶️  正在执行任务 [47/120]: caltech-101 | sepfpl_time_adaptive | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.01 1 40 exp2 '[47/120]'

echo '▶️  正在执行任务 [48/120]: caltech-101 | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 40 exp2 '[48/120]'

echo '▶️  正在执行任务 [49/120]: caltech-101 | dpfpl | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 16 0.4 1 40 exp2 '[49/120]'

echo '▶️  正在执行任务 [50/120]: caltech-101 | sepfpl_hcse | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 16 0.4 1 40 exp2 '[50/120]'

echo '▶️  正在执行任务 [51/120]: caltech-101 | sepfpl_time_adaptive | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.4 1 40 exp2 '[51/120]'

echo '▶️  正在执行任务 [52/120]: caltech-101 | sepfpl | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.4 1 40 exp2 '[52/120]'

echo '▶️  正在执行任务 [53/120]: caltech-101 | dpfpl | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 16 0.1 1 40 exp2 '[53/120]'

echo '▶️  正在执行任务 [54/120]: caltech-101 | sepfpl_hcse | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 16 0.1 1 40 exp2 '[54/120]'

echo '▶️  正在执行任务 [55/120]: caltech-101 | sepfpl_time_adaptive | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.1 1 40 exp2 '[55/120]'

echo '▶️  正在执行任务 [56/120]: caltech-101 | sepfpl | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.1 1 40 exp2 '[56/120]'

echo '▶️  正在执行任务 [57/120]: caltech-101 | dpfpl | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 16 0.01 1 40 exp2 '[57/120]'

echo '▶️  正在执行任务 [58/120]: caltech-101 | sepfpl_hcse | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 16 0.01 1 40 exp2 '[58/120]'

echo '▶️  正在执行任务 [59/120]: caltech-101 | sepfpl_time_adaptive | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.01 1 40 exp2 '[59/120]'

echo '▶️  正在执行任务 [60/120]: caltech-101 | sepfpl | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.01 1 40 exp2 '[60/120]'

echo '▶️  正在执行任务 [61/120]: oxford_pets | dpfpl | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 1 0.4 1 40 exp2 '[61/120]'

echo '▶️  正在执行任务 [62/120]: oxford_pets | sepfpl_hcse | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 1 0.4 1 40 exp2 '[62/120]'

echo '▶️  正在执行任务 [63/120]: oxford_pets | sepfpl_time_adaptive | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.4 1 40 exp2 '[63/120]'

echo '▶️  正在执行任务 [64/120]: oxford_pets | sepfpl | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0.4 1 40 exp2 '[64/120]'

echo '▶️  正在执行任务 [65/120]: oxford_pets | dpfpl | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 1 0.1 1 40 exp2 '[65/120]'

echo '▶️  正在执行任务 [66/120]: oxford_pets | sepfpl_hcse | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 1 0.1 1 40 exp2 '[66/120]'

echo '▶️  正在执行任务 [67/120]: oxford_pets | sepfpl_time_adaptive | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.1 1 40 exp2 '[67/120]'

echo '▶️  正在执行任务 [68/120]: oxford_pets | sepfpl | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0.1 1 40 exp2 '[68/120]'

echo '▶️  正在执行任务 [69/120]: oxford_pets | dpfpl | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 1 0.01 1 40 exp2 '[69/120]'

echo '▶️  正在执行任务 [70/120]: oxford_pets | sepfpl_hcse | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 1 0.01 1 40 exp2 '[70/120]'

echo '▶️  正在执行任务 [71/120]: oxford_pets | sepfpl_time_adaptive | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.01 1 40 exp2 '[71/120]'

echo '▶️  正在执行任务 [72/120]: oxford_pets | sepfpl | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0.01 1 40 exp2 '[72/120]'

echo '▶️  正在执行任务 [73/120]: oxford_pets | dpfpl | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 2 0.4 1 40 exp2 '[73/120]'

echo '▶️  正在执行任务 [74/120]: oxford_pets | sepfpl_hcse | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 2 0.4 1 40 exp2 '[74/120]'

echo '▶️  正在执行任务 [75/120]: oxford_pets | sepfpl_time_adaptive | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.4 1 40 exp2 '[75/120]'

echo '▶️  正在执行任务 [76/120]: oxford_pets | sepfpl | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0.4 1 40 exp2 '[76/120]'

echo '▶️  正在执行任务 [77/120]: oxford_pets | dpfpl | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 2 0.1 1 40 exp2 '[77/120]'

echo '▶️  正在执行任务 [78/120]: oxford_pets | sepfpl_hcse | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 2 0.1 1 40 exp2 '[78/120]'

echo '▶️  正在执行任务 [79/120]: oxford_pets | sepfpl_time_adaptive | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.1 1 40 exp2 '[79/120]'

echo '▶️  正在执行任务 [80/120]: oxford_pets | sepfpl | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0.1 1 40 exp2 '[80/120]'

echo '▶️  正在执行任务 [81/120]: oxford_pets | dpfpl | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 2 0.01 1 40 exp2 '[81/120]'

echo '▶️  正在执行任务 [82/120]: oxford_pets | sepfpl_hcse | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 2 0.01 1 40 exp2 '[82/120]'

echo '▶️  正在执行任务 [83/120]: oxford_pets | sepfpl_time_adaptive | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.01 1 40 exp2 '[83/120]'

echo '▶️  正在执行任务 [84/120]: oxford_pets | sepfpl | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0.01 1 40 exp2 '[84/120]'

echo '▶️  正在执行任务 [85/120]: oxford_pets | dpfpl | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 4 0.4 1 40 exp2 '[85/120]'

echo '▶️  正在执行任务 [86/120]: oxford_pets | sepfpl_hcse | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 4 0.4 1 40 exp2 '[86/120]'

echo '▶️  正在执行任务 [87/120]: oxford_pets | sepfpl_time_adaptive | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.4 1 40 exp2 '[87/120]'

echo '▶️  正在执行任务 [88/120]: oxford_pets | sepfpl | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.4 1 40 exp2 '[88/120]'

echo '▶️  正在执行任务 [89/120]: oxford_pets | dpfpl | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 4 0.1 1 40 exp2 '[89/120]'

echo '▶️  正在执行任务 [90/120]: oxford_pets | sepfpl_hcse | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 4 0.1 1 40 exp2 '[90/120]'

echo '▶️  正在执行任务 [91/120]: oxford_pets | sepfpl_time_adaptive | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.1 1 40 exp2 '[91/120]'

echo '▶️  正在执行任务 [92/120]: oxford_pets | sepfpl | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.1 1 40 exp2 '[92/120]'

echo '▶️  正在执行任务 [93/120]: oxford_pets | dpfpl | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 4 0.01 1 40 exp2 '[93/120]'

echo '▶️  正在执行任务 [94/120]: oxford_pets | sepfpl_hcse | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 4 0.01 1 40 exp2 '[94/120]'

echo '▶️  正在执行任务 [95/120]: oxford_pets | sepfpl_time_adaptive | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.01 1 40 exp2 '[95/120]'

echo '▶️  正在执行任务 [96/120]: oxford_pets | sepfpl | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.01 1 40 exp2 '[96/120]'

echo '▶️  正在执行任务 [97/120]: oxford_pets | dpfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.4 1 40 exp2 '[97/120]'

echo '▶️  正在执行任务 [98/120]: oxford_pets | sepfpl_hcse | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.4 1 40 exp2 '[98/120]'

echo '▶️  正在执行任务 [99/120]: oxford_pets | sepfpl_time_adaptive | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.4 1 40 exp2 '[99/120]'

echo '▶️  正在执行任务 [100/120]: oxford_pets | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 40 exp2 '[100/120]'

echo '▶️  正在执行任务 [101/120]: oxford_pets | dpfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.1 1 40 exp2 '[101/120]'

echo '▶️  正在执行任务 [102/120]: oxford_pets | sepfpl_hcse | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.1 1 40 exp2 '[102/120]'

echo '▶️  正在执行任务 [103/120]: oxford_pets | sepfpl_time_adaptive | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.1 1 40 exp2 '[103/120]'

echo '▶️  正在执行任务 [104/120]: oxford_pets | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 40 exp2 '[104/120]'

echo '▶️  正在执行任务 [105/120]: oxford_pets | dpfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.01 1 40 exp2 '[105/120]'

echo '▶️  正在执行任务 [106/120]: oxford_pets | sepfpl_hcse | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.01 1 40 exp2 '[106/120]'

echo '▶️  正在执行任务 [107/120]: oxford_pets | sepfpl_time_adaptive | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.01 1 40 exp2 '[107/120]'

echo '▶️  正在执行任务 [108/120]: oxford_pets | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 40 exp2 '[108/120]'

echo '▶️  正在执行任务 [109/120]: oxford_pets | dpfpl | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 16 0.4 1 40 exp2 '[109/120]'

echo '▶️  正在执行任务 [110/120]: oxford_pets | sepfpl_hcse | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 16 0.4 1 40 exp2 '[110/120]'

echo '▶️  正在执行任务 [111/120]: oxford_pets | sepfpl_time_adaptive | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.4 1 40 exp2 '[111/120]'

echo '▶️  正在执行任务 [112/120]: oxford_pets | sepfpl | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.4 1 40 exp2 '[112/120]'

echo '▶️  正在执行任务 [113/120]: oxford_pets | dpfpl | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 16 0.1 1 40 exp2 '[113/120]'

echo '▶️  正在执行任务 [114/120]: oxford_pets | sepfpl_hcse | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 16 0.1 1 40 exp2 '[114/120]'

echo '▶️  正在执行任务 [115/120]: oxford_pets | sepfpl_time_adaptive | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.1 1 40 exp2 '[115/120]'

echo '▶️  正在执行任务 [116/120]: oxford_pets | sepfpl | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.1 1 40 exp2 '[116/120]'

echo '▶️  正在执行任务 [117/120]: oxford_pets | dpfpl | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 16 0.01 1 40 exp2 '[117/120]'

echo '▶️  正在执行任务 [118/120]: oxford_pets | sepfpl_hcse | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 16 0.01 1 40 exp2 '[118/120]'

echo '▶️  正在执行任务 [119/120]: oxford_pets | sepfpl_time_adaptive | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.01 1 40 exp2 '[119/120]'

echo '▶️  正在执行任务 [120/120]: oxford_pets | sepfpl | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.01 1 40 exp2 '[120/120]'

