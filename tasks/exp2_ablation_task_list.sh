#!/bin/bash
# Total tasks: 16

# Task 1/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 30 --task-id "[1/16]"

# Task 2/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.0 1 30 --task-id "[2/16]"

# Task 3/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.0 1 30 --task-id "[3/16]"

# Task 4/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.0 1 30 --task-id "[4/16]"

# Task 5/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 30 --task-id "[5/16]"

# Task 6/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.1 1 30 --task-id "[6/16]"

# Task 7/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_hcse 8 0.1 1 30 --task-id "[7/16]"

# Task 8/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 --task-id "[8/16]"

# Task 9/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 1 30 --task-id "[9/16]"

# Task 10/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.0 1 30 --task-id "[10/16]"

# Task 11/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.0 1 30 --task-id "[11/16]"

# Task 12/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.0 1 30 --task-id "[12/16]"

# Task 13/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 30 --task-id "[13/16]"

# Task 14/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.1 1 30 --task-id "[14/16]"

# Task 15/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_hcse 8 0.1 1 30 --task-id "[15/16]"

# Task 16/16
bash srun_main.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl_time_adaptive 8 0.1 1 30 --task-id "[16/16]"

