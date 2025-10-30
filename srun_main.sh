#!/bin/bash

python federated_main.py --root $1 --dataset-config-file $2 --num-users $3 --factorization $4 --rank $5 --noise $6 --seed $7
