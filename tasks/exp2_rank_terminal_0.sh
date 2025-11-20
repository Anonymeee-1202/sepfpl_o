#!/bin/bash
# Terminal 1 tasks (GPU 0) - Total: 18 tasks

# Task 1/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.4 1 30 --task-id "[1/18]"

# Task 2/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 30 --task-id "[2/18]"

# Task 3/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.4 1 30 --task-id "[3/18]"

# Task 4/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.1 1 30 --task-id "[4/18]"

# Task 5/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 30 --task-id "[5/18]"

# Task 6/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.1 1 30 --task-id "[6/18]"

# Task 7/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 4 0.01 1 30 --task-id "[7/18]"

# Task 8/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 30 --task-id "[8/18]"

# Task 9/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 16 0.01 1 30 --task-id "[9/18]"

# Task 10/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.4 1 30 --task-id "[10/18]"

# Task 11/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 30 --task-id "[11/18]"

# Task 12/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.4 1 30 --task-id "[12/18]"

# Task 13/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.1 1 30 --task-id "[13/18]"

# Task 14/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 30 --task-id "[14/18]"

# Task 15/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.1 1 30 --task-id "[15/18]"

# Task 16/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 4 0.01 1 30 --task-id "[16/18]"

# Task 17/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 30 --task-id "[17/18]"

# Task 18/18
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp2_rank WANDB_TAGS=experiment:exp2_rank,type:ablation WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp2_rank bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 16 0.01 1 30 --task-id "[18/18]"

