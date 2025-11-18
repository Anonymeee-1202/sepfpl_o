#!/bin/bash
# Terminal 2 tasks (GPU 1) - Total: 12 tasks

# Task 2/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 noniid-labeldir 30 1

# Task 4/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 noniid-labeldir 30 1

# Task 6/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 noniid-labeldir 30 1

# Task 8/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 noniid-labeldir 30 1

# Task 10/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 noniid-labeldir 30 1

# Task 12/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 noniid-labeldir 30 1

# Task 14/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.4 1 noniid-labeldir 30 1

# Task 16/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.1 1 noniid-labeldir 30 1

# Task 18/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.01 1 noniid-labeldir 30 1

# Task 20/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.4 1 noniid-labeldir 30 1

# Task 22/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.1 1 noniid-labeldir 30 1

# Task 24/24
CUDA_VISIBLE_DEVICES=1 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.01 1 noniid-labeldir 30 1

