#!/bin/bash
# Auto-generated on 2025-11-13 12:12:52
# Total incomplete tasks: 13

export CUDA_VISIBLE_DEVICES=1

# Task  1/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.05 1 noniid-labeldir 30

# Task  2/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.05 1 noniid-labeldir 30

# Task  3/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.05 1 noniid-labeldir 30

# Task  4/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.05 1 noniid-labeldir 30

# Task  5/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.01 1 noniid-labeldir 30

# Task  6/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.01 1 noniid-labeldir 30

# Task  7/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.0 1 noniid-labeldir 30

# Task  8/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.0 1 noniid-labeldir 30

# Task  9/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.4 1 noniid-labeldir 30

# Task 10/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.4 1 noniid-labeldir 30

# Task 11/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.2 1 noniid-labeldir 30

# Task 12/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.2 1 noniid-labeldir 30

# Task 13/13
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.1 1 noniid-labeldir 30

