#!/bin/bash

# 实验任务列表: exp2
# 生成时间: 2025-11-25 11:56:31
# 任务总数: 60
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 顺序执行模式 (无 GPU 或单 GPU)
echo '▶️  正在执行任务 [1/60]: caltech-101 | sepfpl_time_adaptive | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.4 1 30 exp2 '[1/60]'

echo '▶️  正在执行任务 [2/60]: caltech-101 | sepfpl | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.4 1 30 exp2 '[2/60]'

echo '▶️  正在执行任务 [3/60]: caltech-101 | sepfpl_time_adaptive | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.1 1 30 exp2 '[3/60]'

echo '▶️  正在执行任务 [4/60]: caltech-101 | sepfpl | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.1 1 30 exp2 '[4/60]'

echo '▶️  正在执行任务 [5/60]: caltech-101 | sepfpl_time_adaptive | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 1 0.01 1 30 exp2 '[5/60]'

echo '▶️  正在执行任务 [6/60]: caltech-101 | sepfpl | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 1 0.01 1 30 exp2 '[6/60]'

echo '▶️  正在执行任务 [7/60]: caltech-101 | sepfpl_time_adaptive | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.4 1 30 exp2 '[7/60]'

echo '▶️  正在执行任务 [8/60]: caltech-101 | sepfpl | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.4 1 30 exp2 '[8/60]'

echo '▶️  正在执行任务 [9/60]: caltech-101 | sepfpl_time_adaptive | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.1 1 30 exp2 '[9/60]'

echo '▶️  正在执行任务 [10/60]: caltech-101 | sepfpl | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.1 1 30 exp2 '[10/60]'

echo '▶️  正在执行任务 [11/60]: caltech-101 | sepfpl_time_adaptive | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 2 0.01 1 30 exp2 '[11/60]'

echo '▶️  正在执行任务 [12/60]: caltech-101 | sepfpl | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 2 0.01 1 30 exp2 '[12/60]'

echo '▶️  正在执行任务 [13/60]: caltech-101 | sepfpl_time_adaptive | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.4 1 30 exp2 '[13/60]'

echo '▶️  正在执行任务 [14/60]: caltech-101 | sepfpl | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.4 1 30 exp2 '[14/60]'

echo '▶️  正在执行任务 [15/60]: caltech-101 | sepfpl_time_adaptive | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.1 1 30 exp2 '[15/60]'

echo '▶️  正在执行任务 [16/60]: caltech-101 | sepfpl | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.1 1 30 exp2 '[16/60]'

echo '▶️  正在执行任务 [17/60]: caltech-101 | sepfpl_time_adaptive | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 4 0.01 1 30 exp2 '[17/60]'

echo '▶️  正在执行任务 [18/60]: caltech-101 | sepfpl | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.01 1 30 exp2 '[18/60]'

echo '▶️  正在执行任务 [19/60]: caltech-101 | sepfpl_time_adaptive | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.4 1 30 exp2 '[19/60]'

echo '▶️  正在执行任务 [20/60]: caltech-101 | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 30 exp2 '[20/60]'

echo '▶️  正在执行任务 [21/60]: caltech-101 | sepfpl_time_adaptive | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 exp2 '[21/60]'

echo '▶️  正在执行任务 [22/60]: caltech-101 | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 30 exp2 '[22/60]'

echo '▶️  正在执行任务 [23/60]: caltech-101 | sepfpl_time_adaptive | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.01 1 30 exp2 '[23/60]'

echo '▶️  正在执行任务 [24/60]: caltech-101 | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 30 exp2 '[24/60]'

echo '▶️  正在执行任务 [25/60]: caltech-101 | sepfpl_time_adaptive | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.4 1 30 exp2 '[25/60]'

echo '▶️  正在执行任务 [26/60]: caltech-101 | sepfpl | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.4 1 30 exp2 '[26/60]'

echo '▶️  正在执行任务 [27/60]: caltech-101 | sepfpl_time_adaptive | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.1 1 30 exp2 '[27/60]'

echo '▶️  正在执行任务 [28/60]: caltech-101 | sepfpl | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.1 1 30 exp2 '[28/60]'

echo '▶️  正在执行任务 [29/60]: caltech-101 | sepfpl_time_adaptive | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 16 0.01 1 30 exp2 '[29/60]'

echo '▶️  正在执行任务 [30/60]: caltech-101 | sepfpl | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.01 1 30 exp2 '[30/60]'

echo '▶️  正在执行任务 [31/60]: oxford_pets | sepfpl_time_adaptive | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.4 1 30 exp2 '[31/60]'

echo '▶️  正在执行任务 [32/60]: oxford_pets | sepfpl | r=1 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0.4 1 30 exp2 '[32/60]'

echo '▶️  正在执行任务 [33/60]: oxford_pets | sepfpl_time_adaptive | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.1 1 30 exp2 '[33/60]'

echo '▶️  正在执行任务 [34/60]: oxford_pets | sepfpl | r=1 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0.1 1 30 exp2 '[34/60]'

echo '▶️  正在执行任务 [35/60]: oxford_pets | sepfpl_time_adaptive | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 1 0.01 1 30 exp2 '[35/60]'

echo '▶️  正在执行任务 [36/60]: oxford_pets | sepfpl | r=1 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 1 0.01 1 30 exp2 '[36/60]'

echo '▶️  正在执行任务 [37/60]: oxford_pets | sepfpl_time_adaptive | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.4 1 30 exp2 '[37/60]'

echo '▶️  正在执行任务 [38/60]: oxford_pets | sepfpl | r=2 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0.4 1 30 exp2 '[38/60]'

echo '▶️  正在执行任务 [39/60]: oxford_pets | sepfpl_time_adaptive | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.1 1 30 exp2 '[39/60]'

echo '▶️  正在执行任务 [40/60]: oxford_pets | sepfpl | r=2 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0.1 1 30 exp2 '[40/60]'

echo '▶️  正在执行任务 [41/60]: oxford_pets | sepfpl_time_adaptive | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 2 0.01 1 30 exp2 '[41/60]'

echo '▶️  正在执行任务 [42/60]: oxford_pets | sepfpl | r=2 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 2 0.01 1 30 exp2 '[42/60]'

echo '▶️  正在执行任务 [43/60]: oxford_pets | sepfpl_time_adaptive | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.4 1 30 exp2 '[43/60]'

echo '▶️  正在执行任务 [44/60]: oxford_pets | sepfpl | r=4 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.4 1 30 exp2 '[44/60]'

echo '▶️  正在执行任务 [45/60]: oxford_pets | sepfpl_time_adaptive | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.1 1 30 exp2 '[45/60]'

echo '▶️  正在执行任务 [46/60]: oxford_pets | sepfpl | r=4 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.1 1 30 exp2 '[46/60]'

echo '▶️  正在执行任务 [47/60]: oxford_pets | sepfpl_time_adaptive | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 4 0.01 1 30 exp2 '[47/60]'

echo '▶️  正在执行任务 [48/60]: oxford_pets | sepfpl | r=4 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.01 1 30 exp2 '[48/60]'

echo '▶️  正在执行任务 [49/60]: oxford_pets | sepfpl_time_adaptive | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.4 1 30 exp2 '[49/60]'

echo '▶️  正在执行任务 [50/60]: oxford_pets | sepfpl | r=8 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 30 exp2 '[50/60]'

echo '▶️  正在执行任务 [51/60]: oxford_pets | sepfpl_time_adaptive | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 exp2 '[51/60]'

echo '▶️  正在执行任务 [52/60]: oxford_pets | sepfpl | r=8 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 30 exp2 '[52/60]'

echo '▶️  正在执行任务 [53/60]: oxford_pets | sepfpl_time_adaptive | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.01 1 30 exp2 '[53/60]'

echo '▶️  正在执行任务 [54/60]: oxford_pets | sepfpl | r=8 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 30 exp2 '[54/60]'

echo '▶️  正在执行任务 [55/60]: oxford_pets | sepfpl_time_adaptive | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.4 1 30 exp2 '[55/60]'

echo '▶️  正在执行任务 [56/60]: oxford_pets | sepfpl | r=16 n=0.4 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.4 1 30 exp2 '[56/60]'

echo '▶️  正在执行任务 [57/60]: oxford_pets | sepfpl_time_adaptive | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.1 1 30 exp2 '[57/60]'

echo '▶️  正在执行任务 [58/60]: oxford_pets | sepfpl | r=16 n=0.1 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.1 1 30 exp2 '[58/60]'

echo '▶️  正在执行任务 [59/60]: oxford_pets | sepfpl_time_adaptive | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 16 0.01 1 30 exp2 '[59/60]'

echo '▶️  正在执行任务 [60/60]: oxford_pets | sepfpl | r=16 n=0.01 u=10 s=1'
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.01 1 30 exp2 '[60/60]'

