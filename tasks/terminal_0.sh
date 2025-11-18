#!/bin/bash
# Terminal 1 tasks (GPU 0) - Total: 12 tasks

# Task 1/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 noniid-labeldir 30 0

# Task 3/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.2 1 noniid-labeldir 30 0

# Task 5/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.05 1 noniid-labeldir 30 0

# Task 7/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 1 noniid-labeldir 30 0

# Task 9/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.2 1 noniid-labeldir 30 0

# Task 11/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.05 1 noniid-labeldir 30 0

# Task 13/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.0 1 noniid-labeldir 30 0

# Task 15/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.2 1 noniid-labeldir 30 0

# Task 17/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.05 1 noniid-labeldir 30 0

# Task 19/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.0 1 noniid-labeldir 30 0

# Task 21/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.2 1 noniid-labeldir 30 0

# Task 23/24
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/liuxin25/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.05 1 noniid-labeldir 30 0

