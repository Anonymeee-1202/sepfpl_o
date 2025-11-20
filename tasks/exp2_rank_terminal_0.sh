#!/bin/bash
# Terminal 1 tasks (GPU 0) - Total: 18 tasks

# Task 1/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.4 1 30 exp2 "[1/18]"

# Task 2/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 30 exp2 "[2/18]"

# Task 3/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.4 1 30 exp2 "[3/18]"

# Task 4/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.1 1 30 exp2 "[4/18]"

# Task 5/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 30 exp2 "[5/18]"

# Task 6/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.1 1 30 exp2 "[6/18]"

# Task 7/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.01 1 30 exp2 "[7/18]"

# Task 8/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 30 exp2 "[8/18]"

# Task 9/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.01 1 30 exp2 "[9/18]"

# Task 10/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.4 1 30 exp2 "[10/18]"

# Task 11/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 30 exp2 "[11/18]"

# Task 12/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.4 1 30 exp2 "[12/18]"

# Task 13/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.1 1 30 exp2 "[13/18]"

# Task 14/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 30 exp2 "[14/18]"

# Task 15/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.1 1 30 exp2 "[15/18]"

# Task 16/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.01 1 30 exp2 "[16/18]"

# Task 17/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 30 exp2 "[17/18]"

# Task 18/18
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.01 1 30 exp2 "[18/18]"

