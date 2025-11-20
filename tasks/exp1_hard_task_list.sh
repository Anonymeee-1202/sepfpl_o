#!/bin/bash
# Total tasks: 60

# Task 1/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.0 1 30 --task-id "[1/60]"

# Task 2/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.0 1 30 --task-id "[2/60]"

# Task 3/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.0 1 30 --task-id "[3/60]"

# Task 4/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.0 1 30 --task-id "[4/60]"

# Task 5/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.0 1 30 --task-id "[5/60]"

# Task 6/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.4 1 30 --task-id "[6/60]"

# Task 7/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.4 1 30 --task-id "[7/60]"

# Task 8/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.4 1 30 --task-id "[8/60]"

# Task 9/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.4 1 30 --task-id "[9/60]"

# Task 10/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.4 1 30 --task-id "[10/60]"

# Task 11/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.2 1 30 --task-id "[11/60]"

# Task 12/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.2 1 30 --task-id "[12/60]"

# Task 13/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.2 1 30 --task-id "[13/60]"

# Task 14/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.2 1 30 --task-id "[14/60]"

# Task 15/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.2 1 30 --task-id "[15/60]"

# Task 16/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.1 1 30 --task-id "[16/60]"

# Task 17/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.1 1 30 --task-id "[17/60]"

# Task 18/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.1 1 30 --task-id "[18/60]"

# Task 19/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.1 1 30 --task-id "[19/60]"

# Task 20/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.1 1 30 --task-id "[20/60]"

# Task 21/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.05 1 30 --task-id "[21/60]"

# Task 22/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.05 1 30 --task-id "[22/60]"

# Task 23/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.05 1 30 --task-id "[23/60]"

# Task 24/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.05 1 30 --task-id "[24/60]"

# Task 25/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.05 1 30 --task-id "[25/60]"

# Task 26/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 promptfl 8 0.01 1 30 --task-id "[26/60]"

# Task 27/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedotp 8 0.01 1 30 --task-id "[27/60]"

# Task 28/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 fedpgp 8 0.01 1 30 --task-id "[28/60]"

# Task 29/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 dpfpl 8 0.01 1 30 --task-id "[29/60]"

# Task 30/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 25 sepfpl 8 0.01 1 30 --task-id "[30/60]"

# Task 31/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.0 1 30 --task-id "[31/60]"

# Task 32/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.0 1 30 --task-id "[32/60]"

# Task 33/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.0 1 30 --task-id "[33/60]"

# Task 34/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.0 1 30 --task-id "[34/60]"

# Task 35/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.0 1 30 --task-id "[35/60]"

# Task 36/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.4 1 30 --task-id "[36/60]"

# Task 37/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.4 1 30 --task-id "[37/60]"

# Task 38/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.4 1 30 --task-id "[38/60]"

# Task 39/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.4 1 30 --task-id "[39/60]"

# Task 40/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.4 1 30 --task-id "[40/60]"

# Task 41/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.2 1 30 --task-id "[41/60]"

# Task 42/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.2 1 30 --task-id "[42/60]"

# Task 43/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.2 1 30 --task-id "[43/60]"

# Task 44/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.2 1 30 --task-id "[44/60]"

# Task 45/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.2 1 30 --task-id "[45/60]"

# Task 46/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.1 1 30 --task-id "[46/60]"

# Task 47/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.1 1 30 --task-id "[47/60]"

# Task 48/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.1 1 30 --task-id "[48/60]"

# Task 49/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.1 1 30 --task-id "[49/60]"

# Task 50/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.1 1 30 --task-id "[50/60]"

# Task 51/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.05 1 30 --task-id "[51/60]"

# Task 52/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.05 1 30 --task-id "[52/60]"

# Task 53/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.05 1 30 --task-id "[53/60]"

# Task 54/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.05 1 30 --task-id "[54/60]"

# Task 55/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.05 1 30 --task-id "[55/60]"

# Task 56/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 promptfl 8 0.01 1 30 --task-id "[56/60]"

# Task 57/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedotp 8 0.01 1 30 --task-id "[57/60]"

# Task 58/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 fedpgp 8 0.01 1 30 --task-id "[58/60]"

# Task 59/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 dpfpl 8 0.01 1 30 --task-id "[59/60]"

# Task 60/60
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_hard WANDB_TAGS=experiment:exp1_hard,type:hard WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_hard bash srun_main.sh ~/dataset configs/datasets/cifar-100.yaml 50 sepfpl 8 0.01 1 30 --task-id "[60/60]"

