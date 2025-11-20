#!/bin/bash
# Total tasks: 120

# Task 1/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.0 1 30 --task-id "[1/120]"

# Task 2/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.0 1 30 --task-id "[2/120]"

# Task 3/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.0 1 30 --task-id "[3/120]"

# Task 4/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.0 1 30 --task-id "[4/120]"

# Task 5/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 30 --task-id "[5/120]"

# Task 6/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.4 1 30 --task-id "[6/120]"

# Task 7/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.4 1 30 --task-id "[7/120]"

# Task 8/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.4 1 30 --task-id "[8/120]"

# Task 9/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.4 1 30 --task-id "[9/120]"

# Task 10/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 30 --task-id "[10/120]"

# Task 11/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.2 1 30 --task-id "[11/120]"

# Task 12/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.2 1 30 --task-id "[12/120]"

# Task 13/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.2 1 30 --task-id "[13/120]"

# Task 14/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.2 1 30 --task-id "[14/120]"

# Task 15/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.2 1 30 --task-id "[15/120]"

# Task 16/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.1 1 30 --task-id "[16/120]"

# Task 17/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.1 1 30 --task-id "[17/120]"

# Task 18/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.1 1 30 --task-id "[18/120]"

# Task 19/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.1 1 30 --task-id "[19/120]"

# Task 20/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 30 --task-id "[20/120]"

# Task 21/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.05 1 30 --task-id "[21/120]"

# Task 22/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.05 1 30 --task-id "[22/120]"

# Task 23/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.05 1 30 --task-id "[23/120]"

# Task 24/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.05 1 30 --task-id "[24/120]"

# Task 25/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.05 1 30 --task-id "[25/120]"

# Task 26/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 promptfl 8 0.01 1 30 --task-id "[26/120]"

# Task 27/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedotp 8 0.01 1 30 --task-id "[27/120]"

# Task 28/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 fedpgp 8 0.01 1 30 --task-id "[28/120]"

# Task 29/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 dpfpl 8 0.01 1 30 --task-id "[29/120]"

# Task 30/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 30 --task-id "[30/120]"

# Task 31/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.0 1 30 --task-id "[31/120]"

# Task 32/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.0 1 30 --task-id "[32/120]"

# Task 33/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.0 1 30 --task-id "[33/120]"

# Task 34/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.0 1 30 --task-id "[34/120]"

# Task 35/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 1 30 --task-id "[35/120]"

# Task 36/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.4 1 30 --task-id "[36/120]"

# Task 37/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.4 1 30 --task-id "[37/120]"

# Task 38/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.4 1 30 --task-id "[38/120]"

# Task 39/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.4 1 30 --task-id "[39/120]"

# Task 40/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 30 --task-id "[40/120]"

# Task 41/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.2 1 30 --task-id "[41/120]"

# Task 42/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.2 1 30 --task-id "[42/120]"

# Task 43/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.2 1 30 --task-id "[43/120]"

# Task 44/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.2 1 30 --task-id "[44/120]"

# Task 45/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.2 1 30 --task-id "[45/120]"

# Task 46/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.1 1 30 --task-id "[46/120]"

# Task 47/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.1 1 30 --task-id "[47/120]"

# Task 48/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.1 1 30 --task-id "[48/120]"

# Task 49/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.1 1 30 --task-id "[49/120]"

# Task 50/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 30 --task-id "[50/120]"

# Task 51/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.05 1 30 --task-id "[51/120]"

# Task 52/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.05 1 30 --task-id "[52/120]"

# Task 53/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.05 1 30 --task-id "[53/120]"

# Task 54/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.05 1 30 --task-id "[54/120]"

# Task 55/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.05 1 30 --task-id "[55/120]"

# Task 56/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 promptfl 8 0.01 1 30 --task-id "[56/120]"

# Task 57/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedotp 8 0.01 1 30 --task-id "[57/120]"

# Task 58/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 fedpgp 8 0.01 1 30 --task-id "[58/120]"

# Task 59/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 dpfpl 8 0.01 1 30 --task-id "[59/120]"

# Task 60/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 30 --task-id "[60/120]"

# Task 61/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.0 1 30 --task-id "[61/120]"

# Task 62/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.0 1 30 --task-id "[62/120]"

# Task 63/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.0 1 30 --task-id "[63/120]"

# Task 64/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.0 1 30 --task-id "[64/120]"

# Task 65/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.0 1 30 --task-id "[65/120]"

# Task 66/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.4 1 30 --task-id "[66/120]"

# Task 67/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.4 1 30 --task-id "[67/120]"

# Task 68/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.4 1 30 --task-id "[68/120]"

# Task 69/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.4 1 30 --task-id "[69/120]"

# Task 70/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.4 1 30 --task-id "[70/120]"

# Task 71/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.2 1 30 --task-id "[71/120]"

# Task 72/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.2 1 30 --task-id "[72/120]"

# Task 73/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.2 1 30 --task-id "[73/120]"

# Task 74/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.2 1 30 --task-id "[74/120]"

# Task 75/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.2 1 30 --task-id "[75/120]"

# Task 76/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.1 1 30 --task-id "[76/120]"

# Task 77/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.1 1 30 --task-id "[77/120]"

# Task 78/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.1 1 30 --task-id "[78/120]"

# Task 79/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.1 1 30 --task-id "[79/120]"

# Task 80/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.1 1 30 --task-id "[80/120]"

# Task 81/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.05 1 30 --task-id "[81/120]"

# Task 82/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.05 1 30 --task-id "[82/120]"

# Task 83/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.05 1 30 --task-id "[83/120]"

# Task 84/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.05 1 30 --task-id "[84/120]"

# Task 85/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.05 1 30 --task-id "[85/120]"

# Task 86/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 promptfl 8 0.01 1 30 --task-id "[86/120]"

# Task 87/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedotp 8 0.01 1 30 --task-id "[87/120]"

# Task 88/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 fedpgp 8 0.01 1 30 --task-id "[88/120]"

# Task 89/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 dpfpl 8 0.01 1 30 --task-id "[89/120]"

# Task 90/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.01 1 30 --task-id "[90/120]"

# Task 91/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.0 1 30 --task-id "[91/120]"

# Task 92/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.0 1 30 --task-id "[92/120]"

# Task 93/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.0 1 30 --task-id "[93/120]"

# Task 94/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.0 1 30 --task-id "[94/120]"

# Task 95/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.0 1 30 --task-id "[95/120]"

# Task 96/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.4 1 30 --task-id "[96/120]"

# Task 97/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.4 1 30 --task-id "[97/120]"

# Task 98/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.4 1 30 --task-id "[98/120]"

# Task 99/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.4 1 30 --task-id "[99/120]"

# Task 100/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.4 1 30 --task-id "[100/120]"

# Task 101/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.2 1 30 --task-id "[101/120]"

# Task 102/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.2 1 30 --task-id "[102/120]"

# Task 103/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.2 1 30 --task-id "[103/120]"

# Task 104/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.2 1 30 --task-id "[104/120]"

# Task 105/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.2 1 30 --task-id "[105/120]"

# Task 106/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.1 1 30 --task-id "[106/120]"

# Task 107/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.1 1 30 --task-id "[107/120]"

# Task 108/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.1 1 30 --task-id "[108/120]"

# Task 109/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.1 1 30 --task-id "[109/120]"

# Task 110/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.1 1 30 --task-id "[110/120]"

# Task 111/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.05 1 30 --task-id "[111/120]"

# Task 112/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.05 1 30 --task-id "[112/120]"

# Task 113/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.05 1 30 --task-id "[113/120]"

# Task 114/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.05 1 30 --task-id "[114/120]"

# Task 115/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.05 1 30 --task-id "[115/120]"

# Task 116/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 promptfl 8 0.01 1 30 --task-id "[116/120]"

# Task 117/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedotp 8 0.01 1 30 --task-id "[117/120]"

# Task 118/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 fedpgp 8 0.01 1 30 --task-id "[118/120]"

# Task 119/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 dpfpl 8 0.01 1 30 --task-id "[119/120]"

# Task 120/120
WANDB_MODE=online WANDB_PROJECT=dp-fpl WANDB_GROUP=exp1_simple WANDB_TAGS=experiment:exp1_simple,type:simple WANDB_WATCH=gradients WANDB_WATCH_LOGFREQ=200 WANDB_RUN_NAME=exp1_simple bash srun_main.sh ~/dataset configs/datasets/food-101.yaml 10 sepfpl 8 0.01 1 30 --task-id "[120/120]"

