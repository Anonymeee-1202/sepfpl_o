#!/bin/bash
# Terminal 1 tasks (GPU 0) - Total: 16 tasks

# Task 1/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 30 exp2 "[1/16]"

# Task 2/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.0 1 30 exp2 "[2/16]"

# Task 3/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.0 1 30 exp2 "[3/16]"

# Task 4/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.0 1 30 exp2 "[4/16]"

# Task 5/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 30 exp2 "[5/16]"

# Task 6/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.1 1 30 exp2 "[6/16]"

# Task 7/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.1 1 30 exp2 "[7/16]"

# Task 8/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 exp2 "[8/16]"

# Task 9/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 1 30 exp2 "[9/16]"

# Task 10/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.0 1 30 exp2 "[10/16]"

# Task 11/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.0 1 30 exp2 "[11/16]"

# Task 12/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.0 1 30 exp2 "[12/16]"

# Task 13/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 30 exp2 "[13/16]"

# Task 14/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.1 1 30 exp2 "[14/16]"

# Task 15/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.1 1 30 exp2 "[15/16]"

# Task 16/16
CUDA_VISIBLE_DEVICES=0 bash srun_main.sh /home/featurize/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 exp2 "[16/16]"

